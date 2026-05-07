import json
from typing import Any, Dict, Optional
from onesim.models.core.model_manager import ModelManager

class SimpleChatLLM:
    def __init__(self, config_name: str, config_path: str):
        # 初始化模型管理器
        self.model_manager = ModelManager.get_instance()
        self.model_manager.initialize(config_path)

        # 获取模型实例
        self.model = self.model_manager.get_model(config_name=config_name)
        self.model_name = getattr(self.model, "model_name", config_name)

        # client 是对接到具体 API 的
        self.client = getattr(self.model, "client", None)

    def chat(self, user_query: str, system_prompt: str = "You are a helpful assistant", temperature: Optional[float] = None, extra_body: Optional[Dict[str, Any]] = None):
        if not self.client:
            raise RuntimeError("No client available for chat")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        # 调用大模型
        req_extra = {"chat_template_kwargs": {"enable_thinking": False}}
        if isinstance(extra_body, dict):
            # 浅合并，调用方可覆盖默认
            req_extra.update(extra_body)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=(0.9 if temperature is None else float(temperature)),
            extra_body=req_extra,
        )

        return resp.choices[0].message.content

    def chat_json(self, user_query: str, system_prompt: str = "You are a helpful assistant", temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        调用 chat 并尽力解析为 JSON 对象：
        - 首先直接 json.loads
        - 失败时尝试提取第一个 '{' 到最后一个 '}' 的子串再解析
        - 解析仍失败则抛出 ValueError
        """
        content = self.chat(user_query=user_query, system_prompt=system_prompt, temperature=temperature)
        try:
            return json.loads(content)
        except Exception:
            pass

        if isinstance(content, str):
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start : end + 1])
                except Exception:
                    pass
        raise ValueError("LLM 响应并非有效 JSON")

    def choose_analysis_methods(
        self,
        research_paradigm: Optional[str],
        research_question: Optional[str],
        category: Optional[str] = None,
        time_field: Optional[str] = None,
        value_field: Optional[str] = None,
        group_fields: Optional[list] = None,
        allowed_methods: Optional[list] = None,
        alpha: Optional[float] = 0.05,
        max_methods: int = 3,
        temperature: Optional[float] = 0.3,
    ) -> Dict[str, Any]:
        """
        Ask the LLM to select 1-3 statistical methods with minimal, low-bias prompting.
        Returns a compact JSON: {"methods":[{"name":..., "params": {...}}, ...]}.
        """
        if allowed_methods is None:
            allowed_methods = ["group_mean_compare", "robust_ols", "trend_correlation"]

        # Minimal context to reduce bias and over-correction
        context = {
            "research_paradigm": research_paradigm,
            "research_question": research_question,
            "category": category,
            "schema": {
                "time_field": time_field,
                "value_field": value_field,
                "group_fields": group_fields,
            },
            "constraints": {
                "allowed_methods": allowed_methods,
                "alpha": alpha,
                "max_methods": max_methods,
            },
            "instruction": (
                "Select 1-3 statistical methods that directly address the research_question within the given research_paradigm. "
                "Use only names from allowed_methods. Prefer parameters using available schema fields when relevant. "
                "Avoid clustering methods. Return ONLY compact JSON: {'methods': [...]} with no explanations."
            ),
        }

        system_prompt = "Return ONLY valid JSON with a top-level 'methods' list. No explanations."
        user_query = json.dumps(context, ensure_ascii=False)

        try:
            plan = self.chat_json(user_query=user_query, system_prompt=system_prompt, temperature=temperature)
        except Exception:
            plan = None

        # Very light validation and fallback to avoid heavy fix-ups
        methods = []
        if isinstance(plan, dict) and isinstance(plan.get("methods"), list):
            for m in plan.get("methods")[:max_methods]:
                if isinstance(m, dict) and m.get("name") in allowed_methods:
                    params = m.get("params") if isinstance(m.get("params"), dict) else {}
                    methods.append({"name": m["name"], "params": params})

        # Minimal fallback if LLM failed to return usable methods
        if not methods:
            methods = [
                {
                    "name": "trend_correlation",
                    "params": {
                        "x": (time_field or "time"),
                        "y": "endpoint",
                        "method": "spearman",
                    },
                }
            ]

        return {"methods": methods}

if __name__ == "__main__":
    llm = SimpleChatLLM(config_name="openai-gpt4o", config_path="config/model_config.json")
    reply = llm.chat("hello")
    print("模型回复:", reply)
