"""Unified HiGHS solver wrapper for LP solving, IIS extraction, and dual ray extraction."""

import tempfile
import os
from dataclasses import dataclass, field
from pathlib import Path

import highspy
import numpy as np

from dualrayrank.solver.lp_parser import parse_lp_string, parse_lp_file, strip_integrality, LPModel


@dataclass
class SolveResult:
    status: str  # "optimal", "infeasible", "unbounded", "error"
    objective: float | None = None
    message: str = ""
    highs_instance: object = None


@dataclass
class IISResult:
    row_indices: list[int] = field(default_factory=list)
    row_names: list[str] = field(default_factory=list)
    row_texts: list[str] = field(default_factory=list)
    row_bound_status: list[int] = field(default_factory=list)
    col_indices: list[int] = field(default_factory=list)
    success: bool = False


@dataclass
class DualRayResult:
    exists: bool = False
    row_indices: list[int] = field(default_factory=list)
    row_names: list[str] = field(default_factory=list)
    multipliers: list[float] = field(default_factory=list)
    row_texts: list[str] = field(default_factory=list)
    success: bool = False


_STATUS_MAP = {
    highspy.HighsModelStatus.kOptimal: "optimal",
    highspy.HighsModelStatus.kInfeasible: "infeasible",
    highspy.HighsModelStatus.kUnbounded: "unbounded",
    highspy.HighsModelStatus.kObjectiveBound: "optimal",
    highspy.HighsModelStatus.kObjectiveTarget: "optimal",
}


class HiGHSWrapper:
    """Wrapper around HiGHS solver for LP solving and infeasibility diagnostics."""

    def __init__(self, presolve: str = "off", iis_strategy: int = 2, time_limit: float = 60.0):
        self.presolve = presolve
        self.iis_strategy = iis_strategy
        self.time_limit = time_limit

    def _create_solver(self) -> highspy.Highs:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("presolve", self.presolve)
        h.setOptionValue("iis_strategy", self.iis_strategy)
        h.setOptionValue("time_limit", self.time_limit)
        return h

    def _load_model(self, h: highspy.Highs, lp_string_or_path: str) -> str | None:
        """Load LP model from file path or string. Returns temp file path if created, else None."""
        path = lp_string_or_path.strip()
        if os.path.isfile(path):
            status = h.readModel(path)
            if status != highspy.HighsStatus.kOk and status != highspy.HighsStatus.kWarning:
                return None
            return None

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".lp", delete=False)
        tmp.write(lp_string_or_path)
        tmp.close()
        status = h.readModel(tmp.name)
        if status != highspy.HighsStatus.kOk and status != highspy.HighsStatus.kWarning:
            os.unlink(tmp.name)
            return None
        return tmp.name

    def solve_lp(self, lp_string_or_path: str) -> SolveResult:
        """Solve an LP and return status + objective."""
        h = self._create_solver()
        tmp_path = self._load_model(h, lp_string_or_path)
        if tmp_path is None and not os.path.isfile(lp_string_or_path.strip()):
            if h.getNumCol() == 0:
                return SolveResult(status="error", message="Failed to load model")

        try:
            h.run()
            model_status = h.getModelStatus()
            status_str = _STATUS_MAP.get(model_status, "error")
            obj = None
            if status_str == "optimal":
                obj = h.getInfoValue("objective_function_value")[1]
            return SolveResult(
                status=status_str,
                objective=obj,
                message=str(model_status),
                highs_instance=h,
            )
        except Exception as e:
            return SolveResult(status="error", message=str(e))
        finally:
            if tmp_path:
                os.unlink(tmp_path)

    def solve_lp_relaxation(self, lp_string_or_path: str) -> SolveResult:
        """Strip integrality constraints and solve the LP relaxation."""
        path = lp_string_or_path.strip()
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                lp_text = f.read()
        else:
            lp_text = lp_string_or_path

        relaxed = strip_integrality(lp_text)
        return self.solve_lp(relaxed)

    def solve_and_diagnose(self, lp_string_or_path: str) -> tuple[SolveResult, IISResult | None, DualRayResult | None]:
        """Solve LP; if infeasible, also extract IIS and dual ray. Returns (solve_result, iis, dual_ray)."""
        h = self._create_solver()

        path = lp_string_or_path.strip()
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                lp_text = f.read()
        else:
            lp_text = lp_string_or_path

        relaxed = strip_integrality(lp_text)
        lp_model = parse_lp_string(relaxed)

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".lp", delete=False)
        tmp.write(relaxed)
        tmp.close()

        try:
            status = h.readModel(tmp.name)
            if status != highspy.HighsStatus.kOk and status != highspy.HighsStatus.kWarning:
                return SolveResult(status="error", message="Failed to load model"), None, None

            h.run()
            model_status = h.getModelStatus()
            status_str = _STATUS_MAP.get(model_status, "error")

            obj = None
            if status_str == "optimal":
                obj = h.getInfoValue("objective_function_value")[1]

            solve_result = SolveResult(
                status=status_str, objective=obj, message=str(model_status), highs_instance=h
            )

            if status_str != "infeasible":
                return solve_result, None, None

            iis = self.extract_iis(h, lp_model)
            dual_ray = self.extract_dual_ray(h, lp_model)
            return solve_result, iis, dual_ray

        except Exception as e:
            return SolveResult(status="error", message=str(e)), None, None
        finally:
            os.unlink(tmp.name)

    def extract_iis(self, h: highspy.Highs, lp_model: LPModel | None = None) -> IISResult:
        """Extract IIS from an infeasible model. h must have already been solved."""
        result = IISResult()
        try:
            iis_status, iis = h.getIis()
            if iis_status not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
                return result

            row_indices = list(iis.row_index_)
            if not row_indices:
                return result

            result.row_indices = row_indices
            result.success = True

            num_rows = h.getNumRow()
            for idx in row_indices:
                name_status = h.getRowName(idx)
                name = name_status[1] if isinstance(name_status, tuple) else str(name_status)
                result.row_names.append(name)

                if lp_model and 0 <= idx < len(lp_model.constraint_order):
                    cname = lp_model.constraint_order[idx]
                    result.row_texts.append(f"{cname}: {lp_model.constraints[cname]}")
                else:
                    result.row_texts.append(f"row_{idx}")

            result.row_bound_status = [int(b) for b in iis.row_bound_]
            result.col_indices = list(iis.col_index_)
            return result

        except Exception:
            return result

    def extract_dual_ray(self, h: highspy.Highs, lp_model: LPModel | None = None) -> DualRayResult:
        """Extract Farkas dual ray from an infeasible model. h must have already been solved."""
        result = DualRayResult()
        try:
            dr_status, dr_exists, dr_values = h.getDualRay()
            if dr_status not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
                return result
            if not dr_exists:
                return result

            result.exists = True
            result.success = True

            for idx, val in enumerate(dr_values):
                if abs(val) > 1e-12:
                    result.row_indices.append(idx)
                    result.multipliers.append(float(val))

                    name_status = h.getRowName(idx)
                    name = name_status[1] if isinstance(name_status, tuple) else str(name_status)
                    result.row_names.append(name)

                    if lp_model and 0 <= idx < len(lp_model.constraint_order):
                        cname = lp_model.constraint_order[idx]
                        result.row_texts.append(f"{cname}: {lp_model.constraints[cname]}")
                    else:
                        result.row_texts.append(f"row_{idx}")

            return result

        except Exception:
            return result

    @staticmethod
    def check_objective(computed_obj: float, ground_truth_obj: float, tol: float = 1e-6) -> bool:
        """Check if computed objective matches ground truth within tolerance."""
        abs_diff = abs(computed_obj - ground_truth_obj)
        if abs_diff <= 1.0:
            return True
        rel_diff = abs_diff / max(1.0, abs(ground_truth_obj))
        return rel_diff <= tol
