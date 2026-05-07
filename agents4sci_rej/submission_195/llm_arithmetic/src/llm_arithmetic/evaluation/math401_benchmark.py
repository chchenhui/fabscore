"""MATH 401 benchmark implementation based on the paper 'How well do Large Language Models perform in Arithmetic tasks?'."""

import json
import requests
import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from .base_evaluator import ArithmeticProblem


class MATH401Benchmark:
    """Implementation of the MATH 401 benchmark from https://arxiv.org/abs/2304.02015"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent.parent.parent / "data" / "math401"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.problems = []
        self.categories = {
            "euler": 1,
            "add_sub_within_10": 25,
            "add_sub_within_100": 25,
            "add_sub_within_1000": 25,
            "add_sub_within_1e12": 25,
            "add_sub_within_neg10_pos10": 25,
            "decimal_add_sub_within_100": 25,
            "mul_within_100": 25,
            "mul_within_100000": 25,
            "decimal_mul_within_10": 25,
            "div_within_100": 25,
            "exp_int_base_within_10": 25,
            "exp_decimal": 25,
            "irrational_numbers": 25,
            "complex_expressions": 25,
            "trigonometry": 25,
            "logarithms": 25
        }
    
    def download_dataset(self) -> bool:
        """Download the original MATH 401 dataset."""
        url = "https://raw.githubusercontent.com/GanjinZero/math401-llm/main/math401.json"
        cache_file = self.cache_dir / "math401_original.json"
        
        if cache_file.exists():
            print(f"Dataset already cached at {cache_file}")
            return True
        
        try:
            print(f"Downloading MATH 401 dataset from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(cache_file, 'w') as f:
                f.write(response.text)
            
            print(f"Dataset downloaded and cached at {cache_file}")
            return True
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return False
    
    def load_problems(self, force_download: bool = False) -> List[ArithmeticProblem]:
        """Load MATH 401 problems."""
        cache_file = self.cache_dir / "math401_original.json"
        
        # Try to download if not exists or forced
        if not cache_file.exists() or force_download:
            if not self.download_dataset():
                # If download fails, generate a compatible dataset
                return self.generate_math401_compatible()
        
        # Load the dataset
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            problems = []
            for item in data:
                problem = self.parse_math401_item(item)
                if problem:
                    problems.append(problem)
            
            self.problems = problems
            return problems
            
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            # Fall back to generating compatible dataset
            return self.generate_math401_compatible()
    
    def parse_math401_item(self, item: Dict[str, Any]) -> Optional[ArithmeticProblem]:
        """Parse a single MATH 401 dataset item."""
        try:
            # Extract problem and answer
            if isinstance(item, dict):
                problem_text = item.get('question', item.get('problem', ''))
                answer_text = item.get('answer', item.get('response', ''))
            else:
                return None
            
            if not problem_text or not answer_text:
                return None
            
            # Parse answer
            try:
                answer = float(answer_text.strip())
            except ValueError:
                # Try to extract number from answer text
                import re
                numbers = re.findall(r'-?\d+\.?\d*', answer_text)
                if numbers:
                    answer = float(numbers[-1])  # Take last number found
                else:
                    return None
            
            # Categorize the problem
            category = self.categorize_problem(problem_text)
            difficulty = self.determine_difficulty(problem_text, answer)
            operation = self.determine_operation(problem_text)
            operands = self.extract_operands(problem_text)
            
            return ArithmeticProblem(
                problem=problem_text,
                answer=answer,
                operation=operation,
                operands=operands,
                difficulty=difficulty,
                metadata={
                    "category": category,
                    "source": "math401"
                }
            )
            
        except Exception as e:
            print(f"Failed to parse item {item}: {e}")
            return None
    
    def categorize_problem(self, problem: str) -> str:
        """Categorize the problem based on its content."""
        problem_lower = problem.lower()
        
        if 'sin(' in problem_lower or 'cos(' in problem_lower or 'tan(' in problem_lower:
            return "trigonometry"
        elif 'log' in problem_lower:
            return "logarithms"
        elif '**' in problem or '^' in problem:
            return "exponentiation"
        elif 'e' in problem_lower and ('π' in problem or 'pi' in problem_lower):
            return "irrational_numbers"
        elif '(' in problem and ')' in problem:
            return "complex_expressions"
        elif '÷' in problem or '/' in problem:
            return "division"
        elif '×' in problem or '*' in problem:
            return "multiplication"
        elif '+' in problem and '-' in problem:
            return "mixed_operations"
        elif '+' in problem:
            return "addition"
        elif '-' in problem:
            return "subtraction"
        else:
            return "unknown"
    
    def determine_operation(self, problem: str) -> str:
        """Determine the primary operation."""
        if 'sin(' in problem.lower() or 'cos(' in problem.lower() or 'tan(' in problem.lower():
            return "trigonometry"
        elif 'log' in problem.lower():
            return "logarithm"
        elif '**' in problem or '^' in problem:
            return "exponentiation"
        elif '÷' in problem or '/' in problem:
            return "division"
        elif '×' in problem or '*' in problem:
            return "multiplication"
        elif '+' in problem and '-' in problem:
            return "mixed"
        elif '+' in problem:
            return "addition"
        elif '-' in problem:
            return "subtraction"
        else:
            return "other"
    
    def extract_operands(self, problem: str) -> List[float]:
        """Extract numerical operands from the problem."""
        import re
        # Extract numbers (including decimals and negatives)
        numbers = re.findall(r'-?\d+\.?\d*', problem)
        operands = []
        for num_str in numbers:
            try:
                operands.append(float(num_str))
            except ValueError:
                continue
        return operands
    
    def determine_difficulty(self, problem: str, answer: float) -> str:
        """Determine problem difficulty based on complexity."""
        operands = self.extract_operands(problem)
        
        # Check for complex operations
        if any(op in problem.lower() for op in ['sin', 'cos', 'tan', 'log', 'ln']):
            return "hard"
        
        # Check for exponentiation
        if '**' in problem or '^' in problem:
            return "hard"
        
        # Check for complex expressions with brackets
        if '(' in problem and ')' in problem:
            return "medium"
        
        # Check operand sizes
        if operands:
            max_operand = max(abs(op) for op in operands)
            if max_operand > 1000:
                return "hard"
            elif max_operand > 100:
                return "medium"
            else:
                return "easy"
        
        # Check answer magnitude
        if abs(answer) > 1000:
            return "hard"
        elif abs(answer) > 100:
            return "medium"
        else:
            return "easy"
    
    def generate_math401_compatible(self) -> List[ArithmeticProblem]:
        """Generate a MATH 401-compatible dataset if original is unavailable."""
        problems = []
        
        # 1. Euler's equation (e^(iπ) + 1 = 0)
        problems.append(ArithmeticProblem(
            problem="e^(i*π) + 1 =",
            answer=0.0,
            operation="complex",
            operands=[math.e, math.pi, 1],
            difficulty="hard",
            metadata={"category": "euler", "source": "generated"}
        ))
        
        # 2. Addition/Subtraction within different ranges
        import random
        random.seed(42)
        
        # Within 10
        for _ in range(25):
            a, b = random.randint(1, 9), random.randint(1, 9)
            op = random.choice(['+', '-'])
            if op == '+':
                answer = a + b
                operation = "addition"
            else:
                if a < b:
                    a, b = b, a  # Ensure positive result
                answer = a - b
                operation = "subtraction"
            
            problems.append(ArithmeticProblem(
                problem=f"{a} {op} {b} =",
                answer=float(answer),
                operation=operation,
                operands=[float(a), float(b)],
                difficulty="easy",
                metadata={"category": "within_10", "source": "generated"}
            ))
        
        # Within 100
        for _ in range(25):
            a, b = random.randint(10, 99), random.randint(10, 99)
            op = random.choice(['+', '-'])
            if op == '+':
                answer = a + b
                operation = "addition"
            else:
                if a < b:
                    a, b = b, a
                answer = a - b
                operation = "subtraction"
            
            problems.append(ArithmeticProblem(
                problem=f"{a} {op} {b} =",
                answer=float(answer),
                operation=operation,
                operands=[float(a), float(b)],
                difficulty="medium",
                metadata={"category": "within_100", "source": "generated"}
            ))
        
        # Large numbers (within 1e12)
        for _ in range(25):
            a = random.randint(1000000, 999999999999)
            b = random.randint(1000000, 999999999999)
            op = random.choice(['+', '-'])
            if op == '+':
                answer = a + b
                operation = "addition"
            else:
                if a < b:
                    a, b = b, a
                answer = a - b
                operation = "subtraction"
            
            problems.append(ArithmeticProblem(
                problem=f"{a} {op} {b} =",
                answer=float(answer),
                operation=operation,
                operands=[float(a), float(b)],
                difficulty="hard",
                metadata={"category": "large_numbers", "source": "generated"}
            ))
        
        # Decimal operations
        for _ in range(25):
            a = round(random.uniform(-100, 100), 2)
            b = round(random.uniform(-100, 100), 2)
            op = random.choice(['+', '-'])
            if op == '+':
                answer = a + b
                operation = "addition"
            else:
                answer = a - b
                operation = "subtraction"
            
            problems.append(ArithmeticProblem(
                problem=f"{a} {op} {b} =",
                answer=round(answer, 2),
                operation=operation,
                operands=[a, b],
                difficulty="medium",
                metadata={"category": "decimal", "source": "generated"}
            ))
        
        # Multiplication
        for _ in range(25):
            a = random.randint(2, 99)
            b = random.randint(2, 99)
            answer = a * b
            
            problems.append(ArithmeticProblem(
                problem=f"{a} × {b} =",
                answer=float(answer),
                operation="multiplication",
                operands=[float(a), float(b)],
                difficulty="medium",
                metadata={"category": "multiplication", "source": "generated"}
            ))
        
        # Division
        for _ in range(25):
            b = random.randint(2, 20)
            answer = random.randint(2, 50)
            a = b * answer  # Ensure integer result
            
            problems.append(ArithmeticProblem(
                problem=f"{a} ÷ {b} =",
                answer=float(answer),
                operation="division",
                operands=[float(a), float(b)],
                difficulty="medium",
                metadata={"category": "division", "source": "generated"}
            ))
        
        # Exponentiation
        for _ in range(25):
            base = random.randint(2, 9)
            exp = random.randint(2, 4)
            answer = base ** exp
            
            problems.append(ArithmeticProblem(
                problem=f"{base}^{exp} =",
                answer=float(answer),
                operation="exponentiation",
                operands=[float(base), float(exp)],
                difficulty="hard",
                metadata={"category": "exponentiation", "source": "generated"}
            ))
        
        # Trigonometry (basic values)
        trig_problems = [
            ("sin(0)", 0.0),
            ("sin(π/2)", 1.0),
            ("sin(π)", 0.0),
            ("sin(3π/2)", -1.0),
            ("cos(0)", 1.0),
            ("cos(π/2)", 0.0),
            ("cos(π)", -1.0),
            ("cos(3π/2)", 0.0),
            ("tan(0)", 0.0),
            ("tan(π/4)", 1.0),
        ]
        
        for problem_text, answer in trig_problems[:25]:
            problems.append(ArithmeticProblem(
                problem=f"{problem_text} =",
                answer=answer,
                operation="trigonometry",
                operands=[],
                difficulty="hard",
                metadata={"category": "trigonometry", "source": "generated"}
            ))
        
        # Logarithms
        for _ in range(25):
            base = random.choice([2, math.e, 10])
            if base == 2:
                value = random.choice([2, 4, 8, 16, 32, 64])
                answer = math.log2(value)
                problem_text = f"log₂({value})"
            elif base == math.e:
                value = random.choice([1, math.e, math.e**2, math.e**3])
                answer = math.log(value)
                problem_text = f"ln({value:.2f})" if value != 1 and value != math.e else f"ln({int(value) if value == 1 else 'e'})"
            else:  # base 10
                value = random.choice([1, 10, 100, 1000])
                answer = math.log10(value)
                problem_text = f"log({value})"
            
            problems.append(ArithmeticProblem(
                problem=f"{problem_text} =",
                answer=round(answer, 4),
                operation="logarithm",
                operands=[float(value)],
                difficulty="hard",
                metadata={"category": "logarithm", "source": "generated"}
            ))
        
        self.problems = problems
        
        # Save generated dataset
        generated_file = self.cache_dir / "math401_generated.json"
        self.save_problems(problems, str(generated_file))
        
        print(f"Generated MATH 401-compatible dataset with {len(problems)} problems")
        return problems
    
    def save_problems(self, problems: List[ArithmeticProblem], filepath: str) -> None:
        """Save problems to JSON file."""
        data = []
        for problem in problems:
            data.append({
                "problem": problem.problem,
                "answer": problem.answer,
                "operation": problem.operation,
                "operands": problem.operands,
                "difficulty": problem.difficulty,
                "metadata": problem.metadata
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(problems)} problems to {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded problems."""
        if not self.problems:
            return {}
        
        stats = {
            "total_problems": len(self.problems),
            "by_operation": {},
            "by_difficulty": {},
            "by_category": {}
        }
        
        for problem in self.problems:
            # By operation
            op = problem.operation
            if op not in stats["by_operation"]:
                stats["by_operation"][op] = 0
            stats["by_operation"][op] += 1
            
            # By difficulty
            diff = problem.difficulty
            if diff not in stats["by_difficulty"]:
                stats["by_difficulty"][diff] = 0
            stats["by_difficulty"][diff] += 1
            
            # By category
            if problem.metadata and "category" in problem.metadata:
                cat = problem.metadata["category"]
                if cat not in stats["by_category"]:
                    stats["by_category"][cat] = 0
                stats["by_category"][cat] += 1
        
        return stats