"""Base evaluator class for arithmetic reasoning tasks."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import json
import time
from .prompt_configs import PromptType, get_prompt_config


@dataclass
class ArithmeticProblem:
    """Represents a single arithmetic problem."""
    
    problem: str
    answer: float
    operation: str
    operands: List[float]
    difficulty: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Results from evaluating a model on arithmetic problems."""
    
    model_name: str
    total_problems: int
    correct_answers: int
    accuracy: float
    avg_response_time: float
    results_by_operation: Dict[str, Dict[str, float]]
    results_by_difficulty: Dict[str, Dict[str, float]]
    individual_results: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class BaseEvaluator(ABC):
    """Base class for evaluating LLM arithmetic capabilities."""
    
    def __init__(self, model_name: str, verbose: bool = False, prompt_type: PromptType = PromptType.STEP_BY_STEP_BOXED):
        self.model_name = model_name
        self.verbose = verbose
        self.prompt_type = prompt_type
        self.prompt_config = get_prompt_config(prompt_type)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if verbose:
            logging.basicConfig(level=logging.INFO)
    
    @abstractmethod
    def generate_response(self, problem: str) -> str:
        """Generate a response to an arithmetic problem."""
        pass
    
    def extract_answer(self, response: str) -> Optional[float]:
        """Extract numerical answer from model response using prompt-specific pattern only."""
        import re

        # Clean the response - remove commas from numbers
        cleaned_response = response.replace(',', '')

        # Only use the prompt-specific pattern
        prompt_pattern = self.prompt_config.get("answer_pattern")
        if prompt_pattern:
            match = re.search(prompt_pattern, cleaned_response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    pass

        # No fallback - return None if pattern doesn't match
        return None
    
    def evaluate_single_problem(self, problem: ArithmeticProblem) -> Dict[str, Any]:
        """Evaluate a single arithmetic problem."""
        start_time = time.time()
        
        try:
            response = self.generate_response(problem.problem)
            predicted_answer = self.extract_answer(response)
            
            is_correct = (
                predicted_answer is not None and 
                abs(predicted_answer - problem.answer) < 1e-3
            )
            
            response_time = time.time() - start_time
            
            result = {
                "problem": problem.problem,
                "true_answer": problem.answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response": response,
                "response_time": response_time,
                "operation": problem.operation,
                "difficulty": problem.difficulty,
                "operands": problem.operands
            }
            
            if problem.metadata:
                result["metadata"] = problem.metadata
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating problem: {problem.problem}, Error: {e}")
            return {
                "problem": problem.problem,
                "true_answer": problem.answer,
                "predicted_answer": None,
                "is_correct": False,
                "response": f"Error: {str(e)}",
                "response_time": time.time() - start_time,
                "operation": problem.operation,
                "difficulty": problem.difficulty,
                "operands": problem.operands,
                "error": str(e)
            }
    
    def evaluate_problems(self, problems: List[ArithmeticProblem]) -> EvaluationResult:
        """Evaluate a list of arithmetic problems."""
        if self.verbose:
            self.logger.info(f"Evaluating {len(problems)} problems with {self.model_name}")
        
        individual_results = []
        
        for i, problem in enumerate(problems):
            if self.verbose and (i + 1) % 10 == 0:
                self.logger.info(f"Progress: {i + 1}/{len(problems)}")
            
            result = self.evaluate_single_problem(problem)
            individual_results.append(result)
        
        # Compute aggregate statistics
        correct_answers = sum(1 for r in individual_results if r["is_correct"])
        accuracy = correct_answers / len(problems) if problems else 0.0
        avg_response_time = sum(r["response_time"] for r in individual_results) / len(problems) if problems else 0.0
        
        # Compute statistics by operation
        results_by_operation = {}
        for operation in set(p.operation for p in problems):
            op_results = [r for r in individual_results if r["operation"] == operation]
            op_correct = sum(1 for r in op_results if r["is_correct"])
            results_by_operation[operation] = {
                "total": len(op_results),
                "correct": op_correct,
                "accuracy": op_correct / len(op_results) if op_results else 0.0,
                "avg_response_time": sum(r["response_time"] for r in op_results) / len(op_results) if op_results else 0.0
            }
        
        # Compute statistics by difficulty
        results_by_difficulty = {}
        for difficulty in set(p.difficulty for p in problems):
            diff_results = [r for r in individual_results if r["difficulty"] == difficulty]
            diff_correct = sum(1 for r in diff_results if r["is_correct"])
            results_by_difficulty[difficulty] = {
                "total": len(diff_results),
                "correct": diff_correct,
                "accuracy": diff_correct / len(diff_results) if diff_results else 0.0,
                "avg_response_time": sum(r["response_time"] for r in diff_results) / len(diff_results) if diff_results else 0.0
            }
        
        return EvaluationResult(
            model_name=self.model_name,
            total_problems=len(problems),
            correct_answers=correct_answers,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            results_by_operation=results_by_operation,
            results_by_difficulty=results_by_difficulty,
            individual_results=individual_results,
            metadata={
                "prompt_type": self.prompt_type.value,
                "prompt_description": self.prompt_config.get("description", "")
            }
        )
    
    def save_results(self, results: EvaluationResult, filepath: str) -> None:
        """Save evaluation results to a JSON file."""
        results_dict = {
            "model_name": results.model_name,
            "total_problems": results.total_problems,
            "correct_answers": results.correct_answers,
            "accuracy": results.accuracy,
            "avg_response_time": results.avg_response_time,
            "results_by_operation": results.results_by_operation,
            "results_by_difficulty": results.results_by_difficulty,
            "individual_results": results.individual_results,
            "metadata": results.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        if self.verbose:
            self.logger.info(f"Results saved to {filepath}")