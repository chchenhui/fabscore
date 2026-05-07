"""Data preparation utilities for training arithmetic models."""

import json
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from ..evaluation.base_evaluator import ArithmeticProblem
from ..evaluation.benchmark_generator import BenchmarkGenerator


@dataclass
class TrainingExample:
    """A single training example for arithmetic tasks."""
    
    input_text: str
    target_text: str
    problem: ArithmeticProblem
    example_type: str  # "standard", "chain_of_thought", "step_by_step"


class DataPreparator:
    """Prepare training data for arithmetic fine-tuning."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def create_standard_example(self, problem: ArithmeticProblem) -> TrainingExample:
        """Create a standard training example."""
        input_text = f"Solve: {problem.problem}"
        target_text = str(problem.answer)

        return TrainingExample(
            input_text=input_text,
            target_text=target_text,
            problem=problem,
            example_type="standard"
        )

    def create_math401_example(self, problem: ArithmeticProblem) -> TrainingExample:
        """Create a MATH 401-style training example for direct format matching."""
        # Format exactly like MATH 401: "5+9=" -> "14"
        input_text = problem.problem
        if problem.answer == int(problem.answer):
            target_text = str(int(problem.answer))
        else:
            target_text = str(problem.answer)

        return TrainingExample(
            input_text=input_text,
            target_text=target_text,
            problem=problem,
            example_type="math401"
        )
    
    def create_chain_of_thought_example(self, problem: ArithmeticProblem) -> TrainingExample:
        """Create a chain-of-thought training example."""
        input_text = f"Solve step by step: {problem.problem}"
        
        # Generate reasoning steps based on operation
        if problem.operation == "addition":
            if len(problem.operands) == 2:
                a, b = problem.operands
                target_text = f"I need to add {int(a)} + {int(b)}. "
                target_text += f"Adding these numbers: {int(a)} + {int(b)} = {int(problem.answer)}. "
                target_text += f"The answer is {int(problem.answer)}."
            else:
                target_text = f"I need to add {' + '.join(map(str, map(int, problem.operands)))}. "
                running_sum = 0
                steps = []
                for i, operand in enumerate(problem.operands):
                    running_sum += operand
                    if i == 0:
                        steps.append(f"Start with {int(operand)}")
                    else:
                        steps.append(f"Add {int(operand)} to get {int(running_sum)}")
                target_text += " ".join(steps) + f". The answer is {int(problem.answer)}."
        
        elif problem.operation == "subtraction":
            a, b = problem.operands[:2]
            target_text = f"I need to subtract {int(b)} from {int(a)}. "
            target_text += f"Calculating: {int(a)} - {int(b)} = {int(problem.answer)}. "
            target_text += f"The answer is {int(problem.answer)}."
        
        elif problem.operation == "multiplication":
            if len(problem.operands) == 2:
                a, b = problem.operands
                target_text = f"I need to multiply {int(a)} × {int(b)}. "
                target_text += f"Calculating: {int(a)} × {int(b)} = {int(problem.answer)}. "
                target_text += f"The answer is {int(problem.answer)}."
            else:
                target_text = f"I need to multiply {' × '.join(map(str, map(int, problem.operands)))}. "
                running_product = 1
                steps = []
                for i, operand in enumerate(problem.operands):
                    running_product *= operand
                    if i == 0:
                        steps.append(f"Start with {int(operand)}")
                    else:
                        steps.append(f"Multiply by {int(operand)} to get {int(running_product)}")
                target_text += " ".join(steps) + f". The answer is {int(problem.answer)}."
        
        elif problem.operation == "division":
            a, b = problem.operands
            target_text = f"I need to divide {int(a)} by {int(b)}. "
            if problem.answer == int(problem.answer):
                target_text += f"Calculating: {int(a)} ÷ {int(b)} = {int(problem.answer)}. "
                target_text += f"The answer is {int(problem.answer)}."
            else:
                target_text += f"Calculating: {int(a)} ÷ {int(b)} = {problem.answer:.3f}. "
                target_text += f"The answer is {problem.answer:.3f}."

        elif problem.operation == "exponentiation":
            base, exp = problem.operands
            target_text = f"I need to calculate {int(base)} raised to the power of {int(exp)}. "
            target_text += f"This means {int(base)}^{int(exp)} = {int(base)}"
            for i in range(int(exp) - 1):
                target_text += f" × {int(base)}"
            target_text += f" = {int(problem.answer)}. The answer is {int(problem.answer)}."

        elif problem.operation == "logarithm":
            value, base = problem.operands
            if abs(base - 10.0) < 1e-6:
                target_text = f"I need to find log base 10 of {value}. "
                target_text += f"This asks: 10 to what power equals {value}? "
            elif abs(base - 2.718281828459045) < 1e-6:  # e
                target_text = f"I need to find the natural logarithm of {value:.3f}. "
                target_text += f"This asks: e to what power equals {value:.3f}? "
            else:
                target_text = f"I need to find log base {base} of {value}. "
                target_text += f"This asks: {base} to what power equals {value}? "
            target_text += f"The answer is {problem.answer}."

        elif problem.operation == "trigonometry":
            angle_deg = problem.metadata.get("angle_deg", 0)
            func = problem.metadata.get("function", "sin")
            target_text = f"I need to find {func}({angle_deg}°). "
            if func == "sin":
                target_text += f"The sine of {angle_deg} degrees is {problem.answer}."
            elif func == "cos":
                target_text += f"The cosine of {angle_deg} degrees is {problem.answer}."
            elif func == "tan":
                target_text += f"The tangent of {angle_deg} degrees is {problem.answer}."
            target_text += f" The answer is {problem.answer}."

        elif problem.operation == "complex":
            if "euler" in problem.problem.lower():
                target_text = "I need to evaluate Euler's identity: e^(iπ) + 1. "
                target_text += "From Euler's formula, e^(iπ) = cos(π) + i*sin(π) = -1 + 0i = -1. "
                target_text += "Therefore, e^(iπ) + 1 = -1 + 1 = 0. The answer is 0."
            else:
                target_text = f"I need to evaluate this complex number expression. "
                if "i^2" in problem.problem:
                    target_text += "Since i^2 = -1 by definition. The answer is -1."
                elif "i^3" in problem.problem:
                    target_text += "Since i^3 = i^2 × i = -1 × i = -i. Taking the real part: 0."
                elif "i^4" in problem.problem:
                    target_text += "Since i^4 = (i^2)^2 = (-1)^2 = 1. The answer is 1."
                else:
                    target_text += f"The answer is {problem.answer}."
        
        return TrainingExample(
            input_text=input_text,
            target_text=target_text,
            problem=problem,
            example_type="chain_of_thought"
        )
    
    def create_step_by_step_example(self, problem: ArithmeticProblem) -> TrainingExample:
        """Create a detailed step-by-step training example."""
        input_text = f"Solve this problem showing all steps: {problem.problem}"
        
        # Create detailed step-by-step solution
        target_text = f"Problem: {problem.problem}\n"
        target_text += f"Operation: {problem.operation.title()}\n"
        
        if problem.operation == "addition":
            target_text += f"Numbers to add: {', '.join(map(str, map(int, problem.operands)))}\n"
            target_text += f"Calculation: {' + '.join(map(str, map(int, problem.operands)))} = {int(problem.answer)}\n"
            target_text += f"Final answer: {int(problem.answer)}"
        
        elif problem.operation == "subtraction":
            a, b = problem.operands[:2]
            target_text += f"Minuend: {int(a)}\nSubtrahend: {int(b)}\n"
            target_text += f"Calculation: {int(a)} - {int(b)} = {int(problem.answer)}\n"
            target_text += f"Final answer: {int(problem.answer)}"
        
        elif problem.operation == "multiplication":
            target_text += f"Numbers to multiply: {', '.join(map(str, map(int, problem.operands)))}\n"
            target_text += f"Calculation: {' × '.join(map(str, map(int, problem.operands)))} = {int(problem.answer)}\n"
            target_text += f"Final answer: {int(problem.answer)}"
        
        elif problem.operation == "division":
            a, b = problem.operands
            target_text += f"Dividend: {int(a)}\nDivisor: {int(b)}\n"
            if problem.answer == int(problem.answer):
                target_text += f"Calculation: {int(a)} ÷ {int(b)} = {int(problem.answer)}\n"
                target_text += f"Final answer: {int(problem.answer)}"
            else:
                target_text += f"Calculation: {int(a)} ÷ {int(b)} = {problem.answer:.3f}\n"
                target_text += f"Final answer: {problem.answer:.3f}"
        
        return TrainingExample(
            input_text=input_text,
            target_text=target_text,
            problem=problem,
            example_type="step_by_step"
        )
    
    def prepare_training_data(self, problems: List[ArithmeticProblem],
                             example_types: List[str] = None) -> List[TrainingExample]:
        """Prepare training data with different example types."""
        if example_types is None:
            example_types = ["math401", "chain_of_thought"]

        training_examples = []

        for problem in problems:
            # Create examples of each requested type
            for example_type in example_types:
                if example_type == "standard":
                    example = self.create_standard_example(problem)
                elif example_type == "math401":
                    example = self.create_math401_example(problem)
                elif example_type == "chain_of_thought":
                    example = self.create_chain_of_thought_example(problem)
                elif example_type == "step_by_step":
                    example = self.create_step_by_step_example(problem)
                else:
                    continue

                training_examples.append(example)
        
        # Shuffle the examples
        random.shuffle(training_examples)
        
        return training_examples
    
    def save_training_data(self, examples: List[TrainingExample], filepath: str) -> None:
        """Save training examples to a JSON file."""
        data = []
        for example in examples:
            data.append({
                "input_text": example.input_text,
                "target_text": example.target_text,
                "example_type": example.example_type,
                "problem": {
                    "problem": example.problem.problem,
                    "answer": example.problem.answer,
                    "operation": example.problem.operation,
                    "operands": example.problem.operands,
                    "difficulty": example.problem.difficulty,
                    "metadata": example.problem.metadata
                }
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Training data saved to {filepath} with {len(examples)} examples")


class TrainingDatasetGenerator:
    """Generate training datasets for curriculum learning."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.generator = BenchmarkGenerator(seed=seed)
        self.preparator = DataPreparator(seed=seed)
    
    def generate_curriculum_datasets(self, output_dir: Path) -> Dict[str, str]:
        """Generate curriculum learning datasets."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        curriculum_stages = {
            "stage1_easy": {
                "difficulties": ["easy"],
                "operations": ["addition", "subtraction"],
                "count_per_op": 500
            },
            "stage2_medium_basic": {
                "difficulties": ["easy", "medium"],
                "operations": ["addition", "subtraction"],
                "count_per_op": 400
            },
            "stage3_multiplication": {
                "difficulties": ["easy", "medium"],
                "operations": ["multiplication"],
                "count_per_op": 500
            },
            "stage4_division": {
                "difficulties": ["easy", "medium"],
                "operations": ["division"],
                "count_per_op": 500
            },
            "stage5_all_medium": {
                "difficulties": ["medium"],
                "operations": ["addition", "subtraction", "multiplication", "division"],
                "count_per_op": 250
            },
            "stage6_advanced": {
                "difficulties": ["easy", "medium"],
                "operations": ["exponentiation", "logarithm"],
                "count_per_op": 200
            },
            "stage7_specialized": {
                "difficulties": ["easy", "medium"],
                "operations": ["trigonometry", "complex"],
                "count_per_op": 100
            },
            "stage8_hard": {
                "difficulties": ["hard"],
                "operations": ["addition", "subtraction", "multiplication", "division", "exponentiation", "logarithm"],
                "count_per_op": 100
            }
        }
        
        dataset_files = {}
        
        for stage_name, config in curriculum_stages.items():
            problems = []
            
            for operation in config["operations"]:
                for difficulty in config["difficulties"]:
                    if operation == "addition":
                        stage_problems = self.generator.generate_addition_problems(
                            config["count_per_op"], difficulty
                        )
                    elif operation == "subtraction":
                        stage_problems = self.generator.generate_subtraction_problems(
                            config["count_per_op"], difficulty
                        )
                    elif operation == "multiplication":
                        stage_problems = self.generator.generate_multiplication_problems(
                            config["count_per_op"], difficulty
                        )
                    elif operation == "division":
                        stage_problems = self.generator.generate_division_problems(
                            config["count_per_op"], difficulty
                        )
                    elif operation == "exponentiation":
                        stage_problems = self.generator.generate_exponentiation_problems(
                            config["count_per_op"], difficulty
                        )
                    elif operation == "logarithm":
                        stage_problems = self.generator.generate_logarithm_problems(
                            config["count_per_op"], difficulty
                        )
                    elif operation == "trigonometry":
                        stage_problems = self.generator.generate_trigonometry_problems(
                            config["count_per_op"], difficulty
                        )
                    elif operation == "complex":
                        stage_problems = self.generator.generate_complex_problems(
                            config["count_per_op"], difficulty
                        )
                    else:
                        continue
                    
                    problems.extend(stage_problems)
            
            # Create training examples with MATH 401 format + reasoning
            training_examples = self.preparator.prepare_training_data(
                problems,
                example_types=["math401", "chain_of_thought"]
            )
            
            # Save dataset
            dataset_file = output_dir / f"{stage_name}.json"
            self.preparator.save_training_data(training_examples, str(dataset_file))
            dataset_files[stage_name] = str(dataset_file)
        
        return dataset_files
    
    def generate_validation_dataset(self, output_dir: Path) -> str:
        """Generate validation dataset."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate balanced validation set with all operations
        problems = []
        all_operations = ["addition", "subtraction", "multiplication", "division",
                         "exponentiation", "logarithm", "trigonometry", "complex"]

        for operation in all_operations:
            for difficulty in ["easy", "medium", "hard"]:
                # Adjust counts to match MATH 401 proportions
                if operation in ["addition", "subtraction"]:
                    count = 30  # Higher count for basic operations
                elif operation in ["exponentiation", "logarithm", "multiplication", "division"]:
                    count = 20
                elif operation == "trigonometry":
                    count = 10
                elif operation == "complex":
                    count = 5
                else:
                    count = 20

                if operation == "addition":
                    op_problems = self.generator.generate_addition_problems(count, difficulty)
                elif operation == "subtraction":
                    op_problems = self.generator.generate_subtraction_problems(count, difficulty)
                elif operation == "multiplication":
                    op_problems = self.generator.generate_multiplication_problems(count, difficulty)
                elif operation == "division":
                    op_problems = self.generator.generate_division_problems(count, difficulty)
                elif operation == "exponentiation":
                    op_problems = self.generator.generate_exponentiation_problems(count, difficulty)
                elif operation == "logarithm":
                    op_problems = self.generator.generate_logarithm_problems(count, difficulty)
                elif operation == "trigonometry":
                    op_problems = self.generator.generate_trigonometry_problems(count, difficulty)
                elif operation == "complex":
                    op_problems = self.generator.generate_complex_problems(count, difficulty)
                else:
                    continue

                problems.extend(op_problems)
        
        # Create validation examples (MATH 401 format for direct evaluation matching)
        validation_examples = self.preparator.prepare_training_data(
            problems,
            example_types=["math401"]
        )
        
        # Save validation dataset
        validation_file = output_dir / "validation.json"
        self.preparator.save_training_data(validation_examples, str(validation_file))
        
        return str(validation_file)