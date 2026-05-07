import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import argparse

# 相对导入现有代理
from .analyzer_agent import propose_figures as analyzer_propose_figures
from .code_agent import gen_plot_code, patch_code


@dataclass
class Stage1Paths:
    scene_info: Path
    workflow_state: Path
    processed_dir: Optional[Path] = None
    outputs_dir: Path = Path("outputs")


@dataclass
class Stage1Config:
    max_retries: int = 2
    run_timeout_sec: int = 60
    enable_fallback: bool = True


class _Logger:
    def __init__(self, log_path: Path) -> None:
        self._lines: List[str] = []
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        self._write("INFO", msg)

    def info(self, msg: str) -> None:
        self._write("INFO", msg)

    def warning(self, msg: str) -> None:
        self._write("WARNING", msg)

    def error(self, msg: str) -> None:
        self._write("ERROR", msg)

    def debug(self, msg: str) -> None:
        self._write("DEBUG", msg)

    def _write(self, level: str, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            cleaned = msg.rstrip("\n")
        except Exception:
            cleaned = str(msg)
        # 统一格式：YYYY-MM-DD HH:MM:SS | LEVEL    | stage1 - message
        line = f"{ts} | {level:<8} | stage1 - {cleaned}"
        # 控制台输出
        try:
            print(line)
        except Exception:
            pass
        # 追加写入日志文件（实时可见）
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # 仍保留内存缓冲
            self._lines.append(line)

    def flush(self) -> None:
        if not self._lines:
            return
        try:
            with self._path.open("a", encoding="utf-8") as f:
                for line in self._lines:
                    f.write(line + "\n")
        except Exception:
            pass
        self._lines.clear()


def _indent(code: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in code.splitlines())


def _run_python_script(code: str, output_png: Path, timeout_sec: int) -> Tuple[bool, str, str, Path]:
    """
    在临时脚本中运行给定 Python 代码，并尽力保存 matplotlib 最后一幅图到 output_png。
    返回 (success, stdout, stderr, script_path)
    """
    tmp_dir = output_png.parent / "stage1_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / f"run_{output_png.stem}.py"

    wrapper = (
        "import os, sys, traceback\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "OUTPUT_PNG = os.environ.get('STAGE1_OUTPUT_PNG')\n"
        "# monkeypatch show() to auto-save\n"
        "_stage1_saved = {'done': False}\n"
        "try:\n"
        "    import matplotlib.pyplot as plt\n"
        "    def _stage1_mpl_show(*args, **kwargs):\n"
        "        try:\n"
        "            nums = plt.get_fignums()\n"
        "            if nums:\n"
        "                fig = plt.figure(nums[-1])\n"
        "                path = OUTPUT_PNG or 'out.png'\n"
        "                fig.savefig(path, dpi=150, bbox_inches='tight')\n"
        "                _stage1_saved['done'] = True\n"
        "        except Exception:\n"
        "            pass\n"
        "    plt.show = _stage1_mpl_show\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    import plotly.io as pio\n"
        "    def _stage1_plotly_show(fig=None, *args, **kwargs):\n"
        "        try:\n"
        "            if fig is not None and hasattr(fig, 'write_image'):\n"
        "                path = OUTPUT_PNG or 'out.png'\n"
        "                fig.write_image(path, scale=2)\n"
        "                _stage1_saved['done'] = True\n"
        "        except Exception:\n"
        "            pass\n"
        "    pio.show = _stage1_plotly_show\n"
        "except Exception:\n"
        "    pass\n"
        "__stage1_error = None\n"
        "try:\n"
        f"{_indent(code)}\n"
        "except Exception as e:\n"
        "    __stage1_error = e\n"
        "    traceback.print_exc()\n"
        "# try save via matplotlib when not saved\n"
        "saved = bool(_stage1_saved.get('done'))\n"
        "try:\n"
        "    if not saved:\n"
        "        import matplotlib.pyplot as plt\n"
        "        nums = plt.get_fignums()\n"
        "        if nums:\n"
        "            fig = plt.figure(nums[-1])\n"
        "            path = OUTPUT_PNG or 'out.png'\n"
        "            fig.savefig(path, dpi=150, bbox_inches='tight')\n"
        "            saved = True\n"
        "except Exception:\n"
        "    pass\n"
        "if __stage1_error:\n"
        "    sys.exit(1)\n"
        "if not saved and OUTPUT_PNG:\n"
        "    sys.exit(2)\n"
    )

    script_path.write_text(wrapper, encoding="utf-8")

    import subprocess

    env = os.environ.copy()
    env["STAGE1_OUTPUT_PNG"] = str(output_png)
    try:
        cp = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        success = cp.returncode == 0 and output_png.exists()
        return success, cp.stdout, cp.stderr, script_path
    except subprocess.TimeoutExpired as e:  # type: ignore[name-defined]
        return False, e.stdout or "", (e.stderr or "") + "\nTIMEOUT", script_path


def _safe_load_dataset(spec: Dict[str, Any], processed_dir: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
    src = spec.get("source_reference")
    if not src:
        return None
    candidates: List[Path] = []
    try:
        p = Path(src)
        if p.is_file():
            candidates.append(p)
    except Exception:
        pass
    if processed_dir:
        candidates.append(processed_dir / str(src))
    for path in candidates:
        if path.exists() and path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
                data = json.loads(text)
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict)]
                if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                    return [d for d in data["data"] if isinstance(d, dict)]
            except Exception:
                continue
    return None


def _fallback_simple_plot(
    spec: Dict[str, Any],
    output_png: Path,
    csv_out: Path,
    processed_dir: Optional[Path],
) -> Tuple[bool, str]:
    """基于简单聚合绘制折线/柱状作为回退，同时导出 CSV。返回 (success, message)。"""
    data = _safe_load_dataset(spec, processed_dir)
    if not data:
        # 无数据，生成占位图片与空 CSV
        try:
            csv_out.write_text("", encoding="utf-8")
        except Exception:
            pass
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, "No data for fallback", ha="center", va="center")
            plt.axis("off")
            output_png.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_png), dpi=150, bbox_inches="tight")
            plt.close("all")
            return True, "fallback placeholder"
        except Exception as e:
            return False, f"fallback placeholder failed: {e}"

    group_by = spec.get("group_by_fields") or []
    if not isinstance(group_by, list):
        group_by = []
    agg = spec.get("aggregation") or {}
    method = (agg.get("method") or "count").lower()
    field = agg.get("field")

    # 简单聚合：生成 (group_keys..., value)
    table: Dict[Tuple[Any, ...], float] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        key = tuple(row.get(k) for k in group_by) if group_by else ("all",)
        if method == "mean" and field in row:
            # 先累加与计数
            val = float(row.get(field)) if row.get(field) is not None else 0.0
            if key not in table:
                table[key] = (val, 1.0)  # type: ignore[assignment]
            else:
                s, c = table[key]  # type: ignore[misc]
                table[key] = (s + val, c + 1.0)  # type: ignore[assignment]
        elif method == "sum" and field in row:
            val = float(row.get(field)) if row.get(field) is not None else 0.0
            table[key] = table.get(key, 0.0) + val  # type: ignore[assignment]
        else:
            # 默认 count
            table[key] = table.get(key, 0.0) + 1.0  # type: ignore[assignment]

    # 均值后处理
    if method == "mean":
        post: Dict[Tuple[Any, ...], float] = {}
        for k, v in table.items():
            if isinstance(v, tuple) and len(v) == 2:
                s, c = v
                post[k] = (s / max(c, 1.0)) if c else 0.0
            else:
                try:
                    post[k] = float(v)
                except Exception:
                    post[k] = 0.0
        table = post  # type: ignore[assignment]

    # 导出 CSV
    try:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", encoding="utf-8") as f:
            header = list(group_by) + ["value"]
            f.write(",".join(map(str, header)) + "\n")
            for k, v in sorted(table.items(), key=lambda kv: kv[0]):
                row = list(k) + [v]  # type: ignore[list-item]
                f.write(",".join(map(str, row)) + "\n")
    except Exception:
        pass

    # 绘制简单图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_png.parent.mkdir(parents=True, exist_ok=True)
        # 针对 step/time 字段进行数值排序而非字典序
        first_dim = (group_by[0].lower() if group_by and isinstance(group_by[0], str) else None)
        numeric_x = first_dim in {"step", "time", "t", "round"}

        def _num_key_from_value(val: Any):
            if isinstance(val, (int, float)):
                return (0, float(val), str(val))
            s = str(val)
            digits = "".join(ch for ch in s if ch.isdigit())
            if digits:
                try:
                    return (0, float(digits), s)
                except Exception:
                    return (0, 0.0, s)
            # 无数字时，使用次序权重放在数字之后，并以原字符串作为次排序键
            return (1, float("inf"), s)

        if len(group_by) <= 1:
            # 单维度：x 为该维或 'all'
            xs: List[str] = []
            ys: List[float] = []
            if numeric_x:
                sorter = lambda kv: _num_key_from_value(kv[0][0] if isinstance(kv[0], tuple) else kv[0])
            else:
                sorter = lambda kv: kv[0]
            for k, v in sorted(table.items(), key=sorter):
                xs.append(str(k[0]))
                ys.append(float(v))
            vis = (spec.get("suggested_visualization_type") or "line").lower()
            plt.figure(figsize=(6, 4))
            if vis == "bar":
                plt.bar(xs, ys)
            else:
                plt.plot(xs, ys, marker="o")
            plt.title(spec.get("title") or spec.get("id") or "Figure")
            plt.tight_layout()
            plt.savefig(str(output_png), dpi=150)
            plt.close("all")
            return True, "fallback simple 1D"
        else:
            # 双维：按第二维分组，x 为第一维
            if numeric_x:
                dim1 = sorted({k[0] for k in table.keys()}, key=_num_key_from_value)
            else:
                dim1 = sorted({k[0] for k in table.keys()})
            dim2 = sorted({k[1] for k in table.keys()})
            plt.figure(figsize=(7, 4))
            for g in dim2:
                ys: List[float] = []
                for x in dim1:
                    ys.append(float(table.get((x, g), 0.0)))
                plt.plot(list(map(str, dim1)), ys, marker="o", label=str(g))
            plt.legend(title=str(group_by[1]))
            plt.title(spec.get("title") or spec.get("id") or "Figure")
            plt.tight_layout()
            plt.savefig(str(output_png), dpi=150)
            plt.close("all")
            return True, "fallback simple 2D"
    except Exception as e:
        return False, f"fallback plot failed: {e}"


def run_stage1(paths: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> List[Path]:
    """
    编排器 Stage1：
    1) 调用 propose_figures() 产生 3 个 VisualizationSpec
    2) 逐个使用 gen_plot_code() 生成 Python 脚本
    3) run_python 执行脚本，直到得到 3 张 PNG；失败时重试并回退到简版图/CSV 导出

    返回 3 张 PNG 的绝对路径列表。
    """
    # 解析路径与配置
    scene_info = Path(paths.get("scene_info"))
    workflow_state = Path(paths.get("workflow_state"))
    processed_dir = Path(paths.get("processed_dir")) if paths.get("processed_dir") else None
    outputs_dir = Path(paths.get("outputs_dir") or "outputs").absolute()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    conf = Stage1Config(**(cfg or {}))

    logger = _Logger(outputs_dir / "stage1.log")
    logger.info(f"Stage1 start. outputs_dir={outputs_dir}")
    logger.info(f"ENV: ONESIM_MODEL_NAME={os.environ.get('ONESIM_MODEL_NAME')} ONESIM_MODEL_CONFIG={os.environ.get('ONESIM_MODEL_CONFIG')}")
    logger.info(f"scene_info={scene_info}")
    logger.info(f"workflow_state={workflow_state}")
    if processed_dir:
        logger.info(f"processed_dir={processed_dir}")
        # 传递给下游子进程/生成代码，便于在脚本内通过环境变量定位数据集
        try:
            os.environ["STAGE1_PROCESSED_DIR"] = str(processed_dir)
        except Exception:
            pass

    # 产物目标路径
    fig_paths = [outputs_dir / f"fig{i}.png" for i in range(1, 4)]
    csv_paths = [outputs_dir / f"fig{i}.csv" for i in range(1, 4)]
    # 清理旧文件
    for p in fig_paths + csv_paths:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    # 1) 选图
    try:
        specs = analyzer_propose_figures(str(scene_info), str(workflow_state), str(processed_dir) if processed_dir else None)
    except Exception as e:
        logger.log(f"propose_figures failed: {e}")
        specs = []

    if not isinstance(specs, list) or len(specs) < 3:
        logger.warning("propose_figures did not return 3 specs, filling with empty specs")
        while len(specs) < 3:
            specs.append({
                "id": f"fallback_{len(specs)+1}",
                "title": "Fallback Figure",
                "data_source_category": "processed",
                "source_reference": None,
                "group_by_fields": [],
                "aggregation": {"method": "count", "field": None},
                "suggested_visualization_type": "line",
                "why_this_figure": "fallback",
            })

    # 为每个 spec 注入 processed_dir，并尽力解析 source_reference 为绝对数据路径
    try:
        for i in range(len(specs)):
            s = specs[i] if isinstance(specs[i], dict) else {}
            if processed_dir:
                s.setdefault("processed_dir", str(processed_dir))
            src = s.get("source_reference")
            resolved: Optional[Path] = None
            try:
                if isinstance(src, str) and src:
                    p = Path(src)
                    if p.is_file():
                        resolved = p
                    elif processed_dir:
                        cand = processed_dir / src
                        if cand.exists() and cand.is_file():
                            resolved = cand
                if resolved is not None:
                    s["_resolved_data_path"] = str(resolved)
            except Exception:
                pass
            specs[i] = s
    except Exception:
        logger.warning("Failed to enrich specs with processed_dir/_resolved_data_path")

    # 保存选图 JSON（含理由）到 outputs 目录，便于检查
    try:
        plan_path = outputs_dir / "figure_plan.json"
        with plan_path.open("w", encoding="utf-8") as f:
            json.dump(specs, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved figure plan JSON -> {plan_path}")

        for i in range(3):
            try:
                spec_i = specs[i] if i < len(specs) else {}
                spec_path = outputs_dir / f"fig{i+1}_spec.json"
                with spec_path.open("w", encoding="utf-8") as f:
                    json.dump(spec_i, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved fig{i+1} spec -> {spec_path}")
            except Exception as e:
                logger.warning(f"Failed saving fig{i+1} spec: {e}")
    except Exception as e:
        logger.error(f"Failed saving figure plan JSON: {e}")

    # 选图方案摘要
    try:
        for i in range(3):
            it = specs[i] if i < len(specs) else {}
            logger.info(
                f"[plan fig{i+1}] id={it.get('id')} vis={it.get('suggested_visualization_type')} src={it.get('source_reference')} groups={it.get('group_by_fields')}"
            )
    except Exception:
        pass

    # 2) 逐图生成与执行
    for i in range(3):
        spec = specs[i] if i < len(specs) else {}
        fig_path = fig_paths[i]
        csv_path = csv_paths[i]
        logger.info(f"[fig{i+1}] begin: id={spec.get('id')} title={spec.get('title')}")
        try:
            spec_dump_path = outputs_dir / f"fig{i+1}_spec_used.json"
            with spec_dump_path.open("w", encoding="utf-8") as f:
                json.dump(spec, f, ensure_ascii=False, indent=2)
            logger.info(f"[fig{i+1}] wrote spec_used -> {spec_dump_path}")
        except Exception as e:
            logger.warning(f"[fig{i+1}] write spec_used failed: {e}")

        success = False
        last_err_out = ""
        fig_t0 = time.time()

        # 尝试若干次：生成代码 -> 运行 -> 若失败则 patch 一次
        for attempt in range(conf.max_retries + 1):
            logger.info(f"[fig{i+1}] attempt {attempt+1}: generating code")
            try:
                code = gen_plot_code(spec)
            except Exception as e:
                logger.error(f"[fig{i+1}] gen_plot_code failed: {e}")
                code = None
            if not code:
                continue
            try:
                logger.debug(f"[fig{i+1}] attempt {attempt+1}: code length = {len(code)} bytes")
            except Exception:
                pass

            # 保存本次生成的代码到文件
            try:
                code_path = outputs_dir / "stage1_tmp" / f"gen_fig{i+1}_attempt{attempt+1}.py"
                code_path.parent.mkdir(parents=True, exist_ok=True)
                code_path.write_text(code, encoding="utf-8")
                logger.info(f"[fig{i+1}] wrote generated code -> {code_path}")
            except Exception as e:
                logger.warning(f"[fig{i+1}] write generated code failed: {e}")

            ok, so, se, script_path = _run_python_script(code, fig_path, conf.run_timeout_sec)
            # 保存 stdout/stderr
            try:
                (outputs_dir / "stage1_tmp").mkdir(parents=True, exist_ok=True)
                (outputs_dir / "stage1_tmp" / f"run_fig{i+1}_attempt{attempt+1}.stdout.txt").write_text(so or "", encoding="utf-8")
                (outputs_dir / "stage1_tmp" / f"run_fig{i+1}_attempt{attempt+1}.stderr.txt").write_text(se or "", encoding="utf-8")
                logger.debug(f"[fig{i+1}] saved stdout/stderr; script={script_path}")
            except Exception as e:
                logger.warning(f"[fig{i+1}] save stdout/stderr failed: {e}")
            if ok:
                dt = time.time() - fig_t0
                logger.info(f"[fig{i+1}] run success on attempt {attempt+1} in {dt:.2f}s")
                success = True
                break

            last_err_out = (se or "")
            logger.warning(f"[fig{i+1}] run failed attempt {attempt+1}, stderr head=\n{last_err_out[:800]}")

            # patch 一次
            logger.info(f"[fig{i+1}] attempt {attempt+1}: patching code")
            try:
                fixed = patch_code(code, last_err_out or "execution failed")
            except Exception as e:
                logger.error(f"[fig{i+1}] patch_code failed: {e}")
                fixed = None
            if fixed:
                # 保存修复后的代码
                try:
                    fixed_path = outputs_dir / "stage1_tmp" / f"patch_fig{i+1}_attempt{attempt+1}.py"
                    fixed_path.write_text(fixed, encoding="utf-8")
                    logger.info(f"[fig{i+1}] wrote patched code -> {fixed_path}")
                except Exception as e:
                    logger.warning(f"[fig{i+1}] write patched code failed: {e}")

                ok2, so2, se2, script_path2 = _run_python_script(fixed, fig_path, conf.run_timeout_sec)
                # 保存 stdout/stderr for patch run
                try:
                    (outputs_dir / "stage1_tmp" / f"run_fig{i+1}_attempt{attempt+1}.patch.stdout.txt").write_text(so2 or "", encoding="utf-8")
                    (outputs_dir / "stage1_tmp" / f"run_fig{i+1}_attempt{attempt+1}.patch.stderr.txt").write_text(se2 or "", encoding="utf-8")
                    logger.debug(f"[fig{i+1}] saved patch stdout/stderr; script={script_path2}")
                except Exception as e:
                    logger.warning(f"[fig{i+1}] save patch stdout/stderr failed: {e}")
                if ok2:
                    dt = time.time() - fig_t0
                    logger.info(f"[fig{i+1}] patched run success on attempt {attempt+1} in {dt:.2f}s")
                    success = True
                    break
                last_err_out = (se2 or "")
                logger.warning(f"[fig{i+1}] patched run failed attempt {attempt+1}, stderr head=\n{last_err_out[:800]}")

        # 回退
        if not success and conf.enable_fallback:
            ok_fb, msg = _fallback_simple_plot(spec, fig_path, csv_path, processed_dir)
            logger.warning(f"[fig{i+1}] fallback: {msg}")
            success = ok_fb and fig_path.exists()

        if not success:
            # 最后兜底：生成占位图
            logger.warning(f"[fig{i+1}] all attempts failed; writing placeholder image")
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                plt.figure(figsize=(6, 3))
                txt = "Generation failed" if not last_err_out else ("Failed:\n" + last_err_out[:300])
                plt.text(0.5, 0.5, txt, ha="center", va="center", wrap=True)
                plt.axis("off")
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
                plt.close("all")
            except Exception as e:
                logger.error(f"[fig{i+1}] placeholder failed: {e}")

        try:
            size_info = fig_path.stat().st_size if fig_path.exists() else 0
        except Exception:
            size_info = 0
        logger.info(f"[fig{i+1}] done -> {fig_path} (size={size_info} bytes)")

    logger.info("Stage1 done.")
    logger.flush()
    return fig_paths


__all__ = ["run_stage1", "Stage1Paths", "Stage1Config"]



def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage1 Orchestrator Test Runner")
    parser.add_argument("--scene-info", dest="scene_info", help="Path to scene_info.json")
    parser.add_argument("--workflow-state", dest="workflow_state", help="Path to workflow_state.json")
    parser.add_argument("--processed-dir", dest="processed_dir", help="Processed data directory", default=None)
    parser.add_argument("--outputs-dir", dest="outputs_dir", help="Outputs directory (default: outputs)", default=os.environ.get("STAGE1_OUTPUTS_DIR", "outputs"))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("STAGE1_MAX_RETRIES", "2")))
    parser.add_argument("--run-timeout-sec", type=int, default=int(os.environ.get("STAGE1_RUN_TIMEOUT", "60")))
    parser.add_argument("--disable-fallback", action="store_true", help="Disable fallback simple plot and CSV export")

    args = parser.parse_args(argv)

    scene_info = args.scene_info or os.environ.get("SCENE_INFO_PATH")
    workflow_state = args.workflow_state or os.environ.get("WORKFLOW_STATE_PATH")
    processed_dir = args.processed_dir or os.environ.get("PROCESSED_DIR")
    outputs_dir = args.outputs_dir

    if not scene_info or not workflow_state:
        sys.stderr.write("--scene-info and --workflow-state are required (or set SCENE_INFO_PATH/WORKFLOW_STATE_PATH)\n")
        return 2

    paths = {
        "scene_info": scene_info,
        "workflow_state": workflow_state,
        "processed_dir": processed_dir,
        "outputs_dir": outputs_dir,
    }
    cfg = {
        "max_retries": args.max_retries,
        "run_timeout_sec": args.run_timeout_sec,
        "enable_fallback": not args.disable_fallback,
    }

    try:
        figs = run_stage1(paths, cfg)
        for p in figs:
            print(str(Path(p).absolute()))
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
