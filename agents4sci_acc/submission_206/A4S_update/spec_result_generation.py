

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import asyncio
from openai import AsyncOpenAI


MODEL = "claude-3-5-sonnet-20241022"
TEMPERATURE = 0
BASEURL = "https://api.anthropic.com/v1/" #"https://api.anthropic.com/v1/" or "" for openai

class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def render_template(template: str, variables: Dict[str, Any]) -> str:

    template = template.replace("{{", "{").replace("}}", "}")

    return template.format_map(SafeDict({k: str(v) for k, v in variables.items()}))

@runtime_checkable
class LLM(Protocol):
    async def generate_text(self, *, prompt: str, model: Optional[str] = None) -> str:
        ...


@dataclass
class GPT5LLM:
    api_key: Optional[str] = None
    default_model: str = MODEL
    base_url: Optional[str] = None
    concurrency: int = 8

    def __post_init__(self):
        API_KEY = ""

        if not API_KEY:
            raise RuntimeError("NO API Key：")
        if self.base_url:
            self.client = AsyncOpenAI(api_key=API_KEY, base_url=self.base_url)
        else:
            self.client = AsyncOpenAI(api_key=API_KEY)
        self._sem = asyncio.Semaphore(self.concurrency)

    async def generate_text(self, *, prompt: str, model: Optional[str] = None) -> str:

        model_name = model or self.default_model


        async def _call():
            return await self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
            )
        if self._sem:
            async with self._sem:
                resp = await _call()
        else:
            resp = await _call()

        return (resp.choices[0].message.content or "").strip()

@dataclass
class AgentTemplates:
    tpl_spec_designer: str = """

    [SYSTEM]

    Create and define an intermediate specification language that you, as the model, consider maximally readable and reliable for your own use in an NL → Spec ↔ NL confirmation → Code pipeline. Use any notation you prefer.

    [INSTRUCTIONS] 
    Make the definition clear enough that another copy of you can (a) write specs in it from arbitrary NL requests and (b) accurately translate the specs back into NL (c) implement code from such specs after user confirmation.

    [OUTPUT]

    Return ONLY the specification.
    
    """
    
    tpl_fm_interviewer: str = """

    [SYSTEM]  
    You are “FMInterviewer”. From the CURRENT_SPEC (in your invented language), the ORIGINAL_NL_REQUIREMENTS, and the FUNCTION_SIGNATURE, ask the *fewest and most critical* questions needed to resolve ambiguities before implementation.  

    [GOAL]  
    Elicit only decisions that materially affect correctness, safety, or user expectations. Default to safe/common assumptions when possible. Avoid asking cosmetic or trivial questions.  

    [OUTPUT STYLE]  
    - Ask at most 5 numbered questions that correspond ONLY to fields still set to `TBD` in CURRENT_SPEC.
    - Each question ends with [default=VALUE]
    - Tag each with [<spec.path>], e.g., [Semantics.relation_rule]

    [INPUTS]  
    - CURRENT_SPEC:  
    {{CURRENT_SPEC}}  

    - ORIGINAL_NL_REQUIREMENTS:  
    {{USER_REQUIREMENTS}}  

    - USER_EXPLANATION (optional, if provided):  
    {{USER_EXPLANATION}}  

    - FUNCTION_SIGNATURE:  
    {{FUNCTION_SIGNATURE}}  



    """

    tpl_spec_instantiator = """
    [SYSTEM] You are “SpecInstantiator”.
    Fill the SCHEMA with concrete values from the ORIGINAL_NL_REQUIREMENTS and FUNCTION_SIGNATURE.
    If a field is unknown, set it to `TBD`.

    [INPUTS]
    SCHEMA:
    {SCHEMA}

    ORIGINAL_NL_REQUIREMENTS:
    {USER_REQUIREMENTS}

    FUNCTION_SIGNATURE:
    {FUNCTION_SIGNATURE}

    [OUTPUT]
    Return ONLY the instantiated spec (same format as SCHEMA).
    """

    
    tpl_spec_applier: str = """

    [SYSTEM]
    You are “SpecApplier”. Update the CURRENT_SPEC to reflect the user’s answers to the FM questions.

    [INSTRUCTIONS]
    - Align each user answer with the internal tags you previously emitted (e.g., [roles.scope], [concurrency.mode]).
    - Rewrite the spec fully in the SAME invented language and style — coherent, correct, and ready for implementation.
    - Preserve all valid prior content; modify only where answers indicate changes.
    - Resolve ambiguities by applying the user’s decisions; if an answer is missing or unclear, fall back to the suggested safe default for that tag.
    - Do NOT output code.


    [INPUTS]
    - CURRENT_SPEC:
    {{CURRENT_SPEC}}
    - Your FM questions (for reference):
    {{LLM_questions}}
    - USER_ANSWERS (free-text, referencing your tags where possible):
    {{USER_ANSWERS}}
    - FUNCTION_SIGNATURE:
    {{FUNCTION_SIGNATURE}}

    [OUTPUT]
    Return ONLY the updated spec (in your invented language).


    """
    
    tpl_fm_confirmer: str = """

    [SYSTEM]
    You are “FMConfirmer”. Summarize the key decisions just made, in clear natural language, suitable for user confirmation.

    [INSTRUCTIONS]
    - 5–7 sentences max.
    - If relevant in the spec, briefly cover: who is allowed to do it; when it is allowed (states/timing); how concurrent edits are resolved; whether repeated requests are idempotent; what is recorded/audited/notified; how errors are communicated; whether policy recalculation or similar rules apply.
    - End with: “Please confirm or tell me what to change. If confirm, simply return CONFIRM, if you want to change, do not write any code, just tell what to change to meet with your requirement.”

    [INPUTS]
    - UPDATED_SPEC (your invented language):
    {{UPDATED_SPEC}}
    - FUNCTION_SIGNATURE:
    {{FUNCTION_SIGNATURE}}

    [OUTPUT]
    A single natural-language paragraph.


    """
    
    tpl_spec_executor: str = """

        [SYSTEM]
        You are “SpecExecutor”. Implement the agreed functionality described by the AGREED_SPEC.
        - Deliverables in ONE response:
        1) The complete implementation using exactly the provided FUNCTION_SIGNATURE (no modifications)
        2) Ensure the implementation fully satisfies AGREED_SPEC; verify before outputting.
        3) No explanations, no markdown fences.

        [INSTRUCTIONS]
        - Target language/runtime: {{TARGET_LANG}} {{RUNTIME_VERSION}}.
        
        [INPUTS]
        - AGREED_SPEC (your invented language):
        {{AGREED_SPEC}}

        - FUNCTION_SIGNATURE:
        {{FUNCTION_SIGNATURE}}

        [OUTPUT]
        Only Code.

    """


    tpl_agent_answerer: str = """

    You requested the following task:
    "{{USER_REQUIREMENTS}}"

    - FUNCTION_SIGNATURE:
    {{FUNCTION_SIGNATURE}}
    

    To make sure everything is clear before implementation, the assistant has drafted a specification and now asks you to clarify some key points.

    Here are the questions:
    {{LLM_questions}}

    Please reflect on your original request above and answer each question in natural language. 
    Do not worry about any internal symbols or notation from the specification — just give plain, clear answers.

    """

    tpl_user_confirmation: str = """

    Here is the proposed Plan:
    {{FINAL_PLAN}}

    Here is the original user request:
    "{{USER_REQUEST}}"

    - FUNCTION_SIGNATURE:
    {{FUNCTION_SIGNATURE}} 

    Please check: Does the Plan fully satisfy the request?

    Please confirm or tell me what to change.
    If confirm, simply return CONFIRM (exactly this single word).
    If you want to change (incorrect implementation), tell what to change to meet your requirement.


    """

    @staticmethod
    def from_dict(data: Dict[str, str]) -> "AgentTemplates":
        return AgentTemplates(
            tpl_spec_designer=data.get("tpl_spec_designer", ""),
            tpl_fm_interviewer=data.get("tpl_fm_interviewer", ""),
            tpl_spec_applier=data.get("tpl_spec_applier", ""),
            tpl_fm_confirmer=data.get("tpl_fm_confirmer", ""),
            tpl_spec_executor=data.get("tpl_spec_executor", ""),
            tpl_agent_answerer=data.get("tpl_agent_answerer", ""),
            tpl_user_confirmation=data.get("tpl_user_confirmation", ""),
        )

    @staticmethod
    def from_json(path: str) -> "AgentTemplates":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AgentTemplates.from_dict(data)

@dataclass
class GlobalSpecStore:
    spec: Optional[str] = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def ensure(self, llm: LLM, templates: AgentTemplates, model: Optional[str] = None) -> str:
        if self.spec is not None:
            return self.spec
        async with self._lock:
            if self.spec is None:
                prompt = render_template(templates.tpl_spec_designer, {})
                self.spec = await llm.generate_text(prompt=prompt, model=model)
        return self.spec


@dataclass
class Orchestrator:
    llm: LLM
    templates: AgentTemplates
    store: GlobalSpecStore
    target_lang: str = "Python"
    runtime_version: str = "3.8"




    # Step1
    async def ensure_global_spec(self, model: Optional[str] = None) -> str:
        return await self.store.ensure(self.llm, self.templates, model=model)
    


    async def initialize_spec(self,spec, nl_request: str, function_signature: str, model: Optional[str] = None) -> str:
        prompt = render_template(
            self.templates.tpl_spec_instantiator,
            {
                "SCHEMA": spec,
                "USER_REQUIREMENTS": nl_request,
                "FUNCTION_SIGNATURE": function_signature,
            },
        )
        return await self.llm.generate_text(prompt=prompt, model=model)

    async def _ask_fm_questions(self, working_spec: str, nl_request: str, function_signature: str, useradditional:str, model: Optional[str] = None) -> str:
        prompt = render_template(
            self.templates.tpl_fm_interviewer,
            {
                "CURRENT_SPEC": working_spec,
                "USER_REQUIREMENTS": nl_request,
                "FUNCTION_SIGNATURE": function_signature,
                "USER_EXPLANATION": useradditional,

            },
        )
        return await self.llm.generate_text(prompt=prompt, model=model)

    async def _ask_user_answers(self, nl_request: str, llm_questions: str, function_signature: str, model: Optional[str] = None) -> str:
        prompt = render_template(
            self.templates.tpl_agent_answerer,
            {
                "USER_REQUIREMENTS": nl_request,
                "LLM_questions": llm_questions,
                "FUNCTION_SIGNATURE": function_signature,
            },
        )
        return await self.llm.generate_text(prompt=prompt, model=model)

    async def _apply_user_answers(self, working_spec: str, fm_questions: str, user_answers: str, function_signature: str, model: Optional[str] = None) -> str:
        prompt = render_template(
            self.templates.tpl_spec_applier,
            {
                "CURRENT_SPEC": working_spec,
                "LLM_questions": fm_questions,
                "USER_ANSWERS": user_answers,
                "FUNCTION_SIGNATURE": function_signature,
            },
        )
        return await self.llm.generate_text(prompt=prompt, model=model)

    async def _summarize_for_confirm(self, updated_spec: str, function_signature: str, model: Optional[str] = None) -> str:
        prompt = render_template(
            self.templates.tpl_fm_confirmer,
            {
                "UPDATED_SPEC": updated_spec,
                "FUNCTION_SIGNATURE": function_signature,
            },
        )
        return await self.llm.generate_text(prompt=prompt, model=model)

    async def _confirm_by_user(self, final_plan: str, nl_request: str, function_signature: str, model: Optional[str] = None) -> str:
        prompt = render_template(
            self.templates.tpl_user_confirmation,
            {
                "FINAL_PLAN": final_plan,
                "USER_REQUEST": nl_request,
                "FUNCTION_SIGNATURE": function_signature,   
            },
        )
        
        return await self.llm.generate_text(prompt=prompt, model=model)

    async def _implement(self, agreed_spec: str, function_signature: str, model: Optional[str] = None) -> str:
        prompt = render_template(
            self.templates.tpl_spec_executor,
            {
                "AGREED_SPEC": agreed_spec,
                "TARGET_LANG": self.target_lang,
                "RUNTIME_VERSION": self.runtime_version,
                "FUNCTION_SIGNATURE": function_signature,
            },
        )
        return await self.llm.generate_text(prompt=prompt, model=model)



    async def run_task_once(
        self,
        entry: Dict[str, Any],
        *,
        initialize_spec_if_needed: bool = True,
        model: Optional[str] = None,
        max_rounds: int = 5, 
    ) -> Dict[str, Any]:

        if initialize_spec_if_needed:
            global_spec = await self.ensure_global_spec(model=model)
        else:
            if self.store.spec is None:
                raise RuntimeError("GlobalSPEC not initialized yet.")
            global_spec = self.store.spec

        working_spec = str(global_spec)         
        nl_request = entry.get("nl_request", "")  
        useradditional = ""
        fm_questions = ""
        summary = ""
        function_signature = entry.get("prompt", "")
   
        working_spec = await self.initialize_spec(working_spec, nl_request, function_signature, model=model)

        for round_idx in range(1, max_rounds + 1):
            fm_questions = await self._ask_fm_questions(
                working_spec, nl_request, function_signature,useradditional=useradditional, model=model
            )

            user_answers = await self._ask_user_answers(
                nl_request, fm_questions, function_signature, model=model
            )

            updated_spec = await self._apply_user_answers(
                working_spec, fm_questions, user_answers, function_signature, model=model
            )

            summary = await self._summarize_for_confirm(
                updated_spec, function_signature, model=model
            )

            user_confirmation = await self._confirm_by_user(
                summary, nl_request, function_signature, model=model
            )

 
            if "CONFIRM" in str(user_confirmation).strip().upper():
                code = await self._implement(updated_spec, function_signature, model=model)
                result = dict(entry)
                result["generated_solution"] = code
                result["generated_solution_Spec"] = updated_spec
                result["rounds"] = round_idx
                return result

            working_spec = str(updated_spec)
            useradditional = str(user_confirmation)


        code = await self._implement(updated_spec, function_signature, model=model)
        result = dict(entry)
        result["generated_solution"] = code
        result["generated_solution_Spec"] = updated_spec
        round_idx =-1
        result["rounds"] = round_idx
        return result



@dataclass
class OrchestratorConfig:
    """

    """
    templates: AgentTemplates
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = MODEL
    target_lang: str = "Python"
    runtime_version: str = "3.8"
    initialize_spec_if_needed: bool = True
    concurrency: int = 8

    def build(self) -> Orchestrator:
        llm = GPT5LLM(api_key=self.api_key, default_model=self.model, base_url=self.base_url,concurrency=self.concurrency)
        store = GlobalSpecStore()
        return Orchestrator(
            llm=llm,
            templates=self.templates,
            store=store,
            target_lang=self.target_lang,
            runtime_version=self.runtime_version,
        )


async def orchestrate_once(
    entry: Dict[str, Any],
    *,
    # user_answers: str,
    # user_reply: str,
    orch: Orchestrator,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:

    if orch.store.spec is None:
        await orch.ensure_global_spec(model=model_override)
    return await orch.run_task_once(
        entry,
        initialize_spec_if_needed=False,
        model=model_override,
    )



def read_jsonl(path: str) -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(json.loads(s))
    return items

def write_jsonl(path: str, items: list[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")

async def async_main():
    templates = AgentTemplates()

    config = OrchestratorConfig(
        templates=templates,
        # api_key=os.getenv("OPENAI_API_KEY"),
        # model=MODEL,
        base_url=BASEURL,
        model=MODEL,
        target_lang="Python",
        runtime_version="3.8",
        initialize_spec_if_needed=True,
        concurrency=10, 
    )

    entries = read_jsonl("./HumanEval_with_nl.jsonl")

    orch = config.build()
    await orch.ensure_global_spec(model=config.model)
    tasks = [
        orchestrate_once(entry, orch=orch, model_override=config.model)
        for entry in entries
    ]
    results: list[Dict[str, Any]] = await asyncio.gather(*tasks)

    write_jsonl(f"./all_spec_{MODEL}.jsonl", results)


if __name__ == "__main__":
    asyncio.run(async_main())