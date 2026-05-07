from typing import List, Dict, Any, Optional

from datasets import load_dataset

from src.data_processing.base_processor import BaseDatasetProcessor


class PythonInstructionsAlpacaProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 2000):
        super().__init__(
            "iamtarun/python_code_instructions_18k_alpaca",
            "python_instructions_alpaca",
            max_samples,
        )

    def get_system_prompt(self) -> str:
        return "You are an expert Python programmer. Write clean, efficient Python code that follows best practices and includes proper documentation."

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        try:
            instruction = example.get("instruction", "").strip()
            input_text = example.get("input", "").strip()
            output = example.get("output", "").strip()

            if instruction and output:
                user_content = instruction
                if input_text:
                    user_content += f"\n\nInput: {input_text}"

                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output},
                ]
        except Exception:
            return None
        return None


class ArjunPythonQAProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 2000):
        super().__init__("Arjun-G-Ravi/Python-codes", "arjun_python_qa", max_samples)

    def get_system_prompt(self) -> str:
        return "You are a Python programming expert. Answer questions about Python and provide working code examples."

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        try:
            question = example.get("question", "").strip()
            code = example.get("code", "").strip()

            if question and code:
                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": code},
                ]
        except Exception:
            return None
        return None


class GSM8KProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 2000):
        super().__init__("openai/gsm8k", "gsm8k_math", max_samples)
        self.config_name = "main"

    def get_system_prompt(self) -> str:
        return "You are an expert at solving grade school math problems. Show your work step by step and provide the final numerical answer."

    def load_raw_dataset(self):
        dataset = load_dataset(self.dataset_name, self.config_name, split="train")
        return dataset

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        try:
            question = example.get("question", "").strip()
            answer = example.get("answer", "").strip()

            if question and answer:
                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
        except Exception:
            return None
        return None


class SQuADProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 2000):
        super().__init__("squad", "squad_qa", max_samples)

    def get_system_prompt(self) -> str:
        return "You are an expert at reading comprehension. Answer questions based on the given context accurately and concisely."

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        try:
            context = example.get("context", "").strip()
            question = example.get("question", "").strip()
            answers = example.get("answers", {})

            if (
                context
                and question
                and answers
                and "text" in answers
                and len(answers["text"]) > 0
            ):
                answer = answers["text"][0].strip()

                if answer:
                    user_content = f"Context: {context}\n\nQuestion: {question}"
                    return [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": answer},
                    ]
        except Exception:
            return None
        return None


class AlpacaProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 2000):
        super().__init__("tatsu-lab/alpaca", "alpaca_instructions", max_samples)

    def get_system_prompt(self) -> str:
        return "You are a helpful assistant. Follow instructions carefully and provide accurate, helpful responses."

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        try:
            instruction = example.get("instruction", "").strip()
            input_text = example.get("input", "").strip()
            output = example.get("output", "").strip()

            if instruction and output:
                user_content = instruction
                if input_text:
                    user_content += f"\n\nInput: {input_text}"

                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output},
                ]
        except Exception:
            return None
        return None


class CodeSearchNetPythonProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 1500):
        super().__init__(
            "Nan-Do/code-search-net-python", "codesearchnet_python", max_samples
        )

    def get_system_prompt(self) -> str:
        return "You are an expert Python developer. Write Python functions with clear documentation and summaries."

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        try:
            func_name = example.get("func_name", "").strip()
            code = example.get("code", "").strip()
            docstring = example.get("docstring", "").strip()

            if func_name and code:
                instruction = f"Write a Python function named '{func_name}'"
                if docstring:
                    instruction += f" that {docstring}"

                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": code},
                ]
        except Exception:
            return None
        return None


class StableCodePythonProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 1500):
        super().__init__(
            "bunyaminergen/Stable-Code-Python-SFT", "stablecode_python", max_samples
        )

    def get_system_prompt(self) -> str:
        return "You are an expert Python programmer. Follow Python coding instructions precisely and provide high-quality solutions."

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        try:
            instruction = example.get("instruction", "").strip()
            output = example.get("output", "").strip()

            if instruction and output:
                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ]
        except Exception:
            return None
        return None


class MSMarcoProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 1500):
        super().__init__("ms_marco", "ms_marco_qa", max_samples)
        self.config_name = "v1.1"

    def get_system_prompt(self) -> str:
        return "You are an expert at answering questions based on web search results. Provide accurate and concise answers."

    def load_raw_dataset(self):
        try:
            dataset = load_dataset(self.dataset_name, self.config_name, split="train")
            return dataset
        except Exception:
            dataset = load_dataset(self.dataset_name, split="train")
            return dataset

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        try:
            passage = "\n".join(example["passages"].get("passage_text"))
            query = example.get("query", "").strip()
            answers = example.get("answers", [])

            if query and answers and len(answers) > 0:
                answer = (
                    answers[0].strip()
                    if isinstance(answers[0], str)
                    else str(answers[0]).strip()
                )

                if answer and answer != "No Answer Present.":
                    return [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": f"Passage: {passage}\n\nQuery: {query}",
                        },
                        {"role": "assistant", "content": answer},
                    ]
        except Exception:
            return None
        return None


class ArxivSummarizationProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 1500):
        super().__init__("ccdv/arxiv-summarization", "arxiv_summarization", max_samples)

    def get_system_prompt(self) -> str:
        return "You are an expert at summarizing academic papers. Create concise, informative summaries that capture key insights."

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        try:
            article = example.get("article", "").strip()
            abstract = example.get("abstract", "").strip()

            if article and abstract and len(article) > 200:  # Ensure meaningful content
                instruction = "Summarize the following academic paper:"
                user_content = f"{instruction}\n\n{article}"

                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": abstract},
                ]
        except Exception:
            return None
        return None


class LatinTranslationProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 1500):
        super().__init__(
            "grosenthal/latin_english_translation", "latin_translation", max_samples
        )

    def get_system_prompt(self) -> str:
        return "You are an expert in Latin translation. Provide accurate translations between Latin and English."

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        try:
            latin = example.get("la", "").strip()
            english = example.get("en", "").strip()

            if latin and english:
                return [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Translate this Latin text to English: {latin}",
                    },
                    {"role": "assistant", "content": english},
                ]
        except Exception:
            pass
        return None


class StyleTransferProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 1500):
        super().__init__(
            "Lots-of-LoRAs/task933_wiki_auto_style_transfer",
            "style_transfer",
            max_samples,
        )

    def get_system_prompt(self) -> str:
        return "You are an expert at text style transfer. Rewrite text to improve clarity and readability."

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        try:
            input_text = example.get("input", "").strip()
            output_list = example.get("output", [])

            if input_text and output_list:
                output_text = output_list[0].strip()

                if output_text:
                    return [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": input_text,
                        },
                        {"role": "assistant", "content": output_text},
                    ]
        except Exception:
            pass
        return None


class IMDBProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 1500):
        super().__init__(
            "stanfordnlp/imdb",
            "imdb_sentiment",
            max_samples,
        )

    def get_system_prompt(self) -> str:
        return "You are an expert at ranking sentiment of movie reviews"

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        text = example["text"]
        label = example["label"]
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Is sentiment positive or negative?\n\n{text}",
            },
            {"role": "assistant", "content": "positive" if label == 1 else "negaitive"},
        ]


class OrcaMathProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 1500):
        super().__init__(
            "microsoft/orca-math-word-problems-200k", "orca_math", max_samples
        )

    def get_system_prompt(self) -> str:
        return "You are an expert mathematician. Solve mathematical word problems with step-by-step reasoning."

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        try:
            question = example.get("question", "").strip()
            answer = example.get("answer", "").strip()

            if question and answer:
                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
        except Exception:
            pass


class XSumProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 1500):
        super().__init__("EdinburghNLP/xsum", "xsum", max_samples)

    def get_system_prompt(self) -> str:
        return "You are a professional writer"

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Compress this document:\n\n{example['document']}",
            },
            {"role": "assistant", "content": example["summary"]},
        ]


class ReasonMathProcessor(BaseDatasetProcessor):
    def __init__(self, max_samples: int = 1500):
        super().__init__("open-r1/OpenThoughts-114k-math", "reason_math", max_samples)

    def get_system_prompt(self) -> str:
        return "You are a professional mathematician"

    def process_example(
        self, example: Dict[str, Any]
    ) -> Optional[List[Dict[str, str]]]:
        messages = example["messages"]
        return [{"role": "system", "content": example["system"]}] + messages
