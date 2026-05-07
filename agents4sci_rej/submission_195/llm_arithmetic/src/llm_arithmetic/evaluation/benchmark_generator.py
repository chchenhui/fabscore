"""Generate arithmetic benchmarks for evaluation."""

import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .base_evaluator import ArithmeticProblem


class BenchmarkGenerator:
    """Generate arithmetic problems for benchmarking."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_addition_problems(self, count: int, difficulty: str = "easy") -> List[ArithmeticProblem]:
        """Generate addition problems."""
        problems = []
        
        if difficulty == "easy":
            # Single-digit or small numbers
            for _ in range(count):
                a = random.randint(1, 99)
                b = random.randint(1, 99)
                answer = a + b
                problem_text = f"{a} + {b}"
                
                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="addition",
                    operands=[float(a), float(b)],
                    difficulty=difficulty,
                    metadata={"digits_a": len(str(a)), "digits_b": len(str(b))}
                ))
        
        elif difficulty == "medium":
            # 2-3 digit numbers
            for _ in range(count):
                a = random.randint(100, 999)
                b = random.randint(100, 999)
                answer = a + b
                problem_text = f"{a} + {b}"
                
                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="addition",
                    operands=[float(a), float(b)],
                    difficulty=difficulty,
                    metadata={"digits_a": len(str(a)), "digits_b": len(str(b))}
                ))
        
        elif difficulty == "hard":
            # Large numbers or multiple operands
            for _ in range(count):
                if random.random() < 0.5:  # Large 2-operand problems
                    a = random.randint(1000, 99999)
                    b = random.randint(1000, 99999)
                    answer = a + b
                    problem_text = f"{a} + {b}"
                    operands = [float(a), float(b)]
                else:  # 3-operand problems
                    a = random.randint(10, 999)
                    b = random.randint(10, 999)
                    c = random.randint(10, 999)
                    answer = a + b + c
                    problem_text = f"{a} + {b} + {c}"
                    operands = [float(a), float(b), float(c)]
                
                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="addition",
                    operands=operands,
                    difficulty=difficulty,
                    metadata={"num_operands": len(operands)}
                ))
        
        return problems
    
    def generate_subtraction_problems(self, count: int, difficulty: str = "easy") -> List[ArithmeticProblem]:
        """Generate subtraction problems."""
        problems = []
        
        if difficulty == "easy":
            for _ in range(count):
                a = random.randint(10, 99)
                b = random.randint(1, a)  # Ensure positive result
                answer = a - b
                problem_text = f"{a} - {b}"
                
                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="subtraction",
                    operands=[float(a), float(b)],
                    difficulty=difficulty,
                    metadata={"digits_a": len(str(a)), "digits_b": len(str(b))}
                ))
        
        elif difficulty == "medium":
            for _ in range(count):
                a = random.randint(100, 999)
                b = random.randint(1, a)  # Ensure positive result
                answer = a - b
                problem_text = f"{a} - {b}"
                
                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="subtraction",
                    operands=[float(a), float(b)],
                    difficulty=difficulty,
                    metadata={"digits_a": len(str(a)), "digits_b": len(str(b))}
                ))
        
        elif difficulty == "hard":
            for _ in range(count):
                # Allow negative results and larger numbers
                a = random.randint(-9999, 99999)
                b = random.randint(-9999, 99999)
                answer = a - b
                problem_text = f"{a} - {b}"
                
                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="subtraction",
                    operands=[float(a), float(b)],
                    difficulty=difficulty,
                    metadata={"negative_result": answer < 0}
                ))
        
        return problems
    
    def generate_multiplication_problems(self, count: int, difficulty: str = "easy") -> List[ArithmeticProblem]:
        """Generate multiplication problems."""
        problems = []
        
        if difficulty == "easy":
            # Single-digit multiplication
            for _ in range(count):
                a = random.randint(1, 12)
                b = random.randint(1, 12)
                answer = a * b
                problem_text = f"{a} × {b}"
                
                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="multiplication",
                    operands=[float(a), float(b)],
                    difficulty=difficulty,
                    metadata={"single_digit": True}
                ))
        
        elif difficulty == "medium":
            # One or two digit numbers
            for _ in range(count):
                a = random.randint(10, 99)
                b = random.randint(1, 99)
                answer = a * b
                problem_text = f"{a} × {b}"
                
                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="multiplication",
                    operands=[float(a), float(b)],
                    difficulty=difficulty,
                    metadata={"digits_a": len(str(a)), "digits_b": len(str(b))}
                ))
        
        elif difficulty == "hard":
            # Large numbers or multiple operands
            for _ in range(count):
                if random.random() < 0.7:  # Large 2-operand problems
                    a = random.randint(100, 999)
                    b = random.randint(100, 999)
                    answer = a * b
                    problem_text = f"{a} × {b}"
                    operands = [float(a), float(b)]
                else:  # 3-operand problems
                    a = random.randint(2, 20)
                    b = random.randint(2, 20)
                    c = random.randint(2, 20)
                    answer = a * b * c
                    problem_text = f"{a} × {b} × {c}"
                    operands = [float(a), float(b), float(c)]
                
                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="multiplication",
                    operands=operands,
                    difficulty=difficulty,
                    metadata={"num_operands": len(operands)}
                ))
        
        return problems
    
    def generate_division_problems(self, count: int, difficulty: str = "easy") -> List[ArithmeticProblem]:
        """Generate division problems with integer results."""
        problems = []
        
        if difficulty == "easy":
            # Simple division with single-digit results
            for _ in range(count):
                quotient = random.randint(1, 12)
                divisor = random.randint(1, 12)
                dividend = quotient * divisor
                answer = quotient
                problem_text = f"{dividend} ÷ {divisor}"
                
                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="division",
                    operands=[float(dividend), float(divisor)],
                    difficulty=difficulty,
                    metadata={"exact_division": True}
                ))
        
        elif difficulty == "medium":
            # Two-digit division
            for _ in range(count):
                quotient = random.randint(10, 99)
                divisor = random.randint(2, 20)
                dividend = quotient * divisor
                answer = quotient
                problem_text = f"{dividend} ÷ {divisor}"
                
                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="division",
                    operands=[float(dividend), float(divisor)],
                    difficulty=difficulty,
                    metadata={"exact_division": True}
                ))
        
        elif difficulty == "hard":
            # Large numbers or decimal results
            for _ in range(count):
                if random.random() < 0.5:  # Exact division with large numbers
                    quotient = random.randint(100, 999)
                    divisor = random.randint(10, 99)
                    dividend = quotient * divisor
                    answer = quotient
                    problem_text = f"{dividend} ÷ {divisor}"
                    exact = True
                else:  # Division with decimal results
                    dividend = random.randint(100, 999)
                    divisor = random.randint(7, 13)
                    answer = dividend / divisor
                    problem_text = f"{dividend} ÷ {divisor}"
                    exact = False
                
                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="division",
                    operands=[float(dividend), float(divisor)],
                    difficulty=difficulty,
                    metadata={"exact_division": exact}
                ))
        
        return problems
    
    def generate_mixed_problems(self, count: int, difficulty: str = "easy") -> List[ArithmeticProblem]:
        """Generate mixed arithmetic problems."""
        problems = []
        operations = ["addition", "subtraction", "multiplication", "division"]
        
        per_operation = count // len(operations)
        remainder = count % len(operations)
        
        for i, operation in enumerate(operations):
            op_count = per_operation + (1 if i < remainder else 0)
            
            if operation == "addition":
                problems.extend(self.generate_addition_problems(op_count, difficulty))
            elif operation == "subtraction":
                problems.extend(self.generate_subtraction_problems(op_count, difficulty))
            elif operation == "multiplication":
                problems.extend(self.generate_multiplication_problems(op_count, difficulty))
            elif operation == "division":
                problems.extend(self.generate_division_problems(op_count, difficulty))
        
        # Shuffle to mix operations
        random.shuffle(problems)
        return problems
    
    def generate_math401_benchmark(self) -> List[ArithmeticProblem]:
        """Generate the MATH 401+ benchmark dataset."""
        problems = []
        
        # Generate problems for each difficulty level
        difficulties = ["easy", "medium", "hard"]
        
        for difficulty in difficulties:
            # 25 problems of each operation type per difficulty
            problems.extend(self.generate_addition_problems(25, difficulty))
            problems.extend(self.generate_subtraction_problems(25, difficulty))
            problems.extend(self.generate_multiplication_problems(25, difficulty))
            problems.extend(self.generate_division_problems(25, difficulty))
        
        # Add some mixed operation problems
        problems.extend(self.generate_mixed_problems(100, "medium"))
        
        # Shuffle the final dataset
        random.shuffle(problems)
        
        return problems

    def generate_exponentiation_problems(self, count: int, difficulty: str = "easy") -> List[ArithmeticProblem]:
        """Generate exponentiation problems."""
        problems = []

        if difficulty == "easy":
            # Small base and exponent
            for _ in range(count):
                base = random.randint(2, 10)
                exponent = random.randint(2, 4)
                answer = base ** exponent
                problem_text = f"{base}^{exponent}"

                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="exponentiation",
                    operands=[float(base), float(exponent)],
                    difficulty=difficulty,
                    metadata={"base": base, "exponent": exponent}
                ))

        elif difficulty == "medium":
            # Moderate base and exponent
            for _ in range(count):
                base = random.randint(2, 15)
                exponent = random.randint(2, 6)
                answer = base ** exponent
                if answer > 10000:  # Keep reasonable size
                    base = random.randint(2, 8)
                    exponent = random.randint(2, 4)
                    answer = base ** exponent
                problem_text = f"{base}^{exponent}"

                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="exponentiation",
                    operands=[float(base), float(exponent)],
                    difficulty=difficulty,
                    metadata={"base": base, "exponent": exponent}
                ))

        elif difficulty == "hard":
            # Larger exponents or special cases
            for _ in range(count):
                if random.random() < 0.3:  # Special cases
                    choices = [(2, 10), (3, 8), (5, 6), (10, 4), (4, 7)]
                    base, exponent = random.choice(choices)
                else:
                    base = random.randint(2, 20)
                    exponent = random.randint(3, 8)

                answer = base ** exponent
                if answer > 100000:  # Keep reasonable
                    base = random.randint(2, 10)
                    exponent = random.randint(3, 5)
                    answer = base ** exponent
                problem_text = f"{base}^{exponent}"

                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="exponentiation",
                    operands=[float(base), float(exponent)],
                    difficulty=difficulty,
                    metadata={"base": base, "exponent": exponent}
                ))

        return problems

    def generate_logarithm_problems(self, count: int, difficulty: str = "easy") -> List[ArithmeticProblem]:
        """Generate logarithm problems."""
        import math
        problems = []

        if difficulty == "easy":
            # Simple log base 10 and natural log
            for _ in range(count):
                if random.random() < 0.5:  # log base 10
                    power = random.randint(1, 4)
                    value = 10 ** power
                    answer = power
                    problem_text = f"log10({value})"
                    operands = [float(value), 10.0]
                else:  # Simple powers of e
                    power = random.randint(1, 3)
                    value = round(math.e ** power, 3)
                    answer = power
                    problem_text = f"ln({value:.3f})"
                    operands = [float(value), math.e]

                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="logarithm",
                    operands=operands,
                    difficulty=difficulty,
                    metadata={"base": operands[1]}
                ))

        elif difficulty in ["medium", "hard"]:
            # Mixed logarithms
            for _ in range(count):
                if random.random() < 0.4:  # log base 2
                    power = random.randint(1, 8)
                    value = 2 ** power
                    answer = power
                    problem_text = f"log2({value})"
                    operands = [float(value), 2.0]
                elif random.random() < 0.7:  # log base 10
                    power = random.randint(1, 5)
                    value = 10 ** power
                    answer = power
                    problem_text = f"log10({value})"
                    operands = [float(value), 10.0]
                else:  # natural log
                    values = [1, math.e, math.e**2, math.e**3]
                    answers = [0, 1, 2, 3]
                    idx = random.randint(0, len(values)-1)
                    value = values[idx]
                    answer = answers[idx]
                    problem_text = f"ln({value:.3f})" if value != 1 else "ln(1)"
                    operands = [float(value), math.e]

                problems.append(ArithmeticProblem(
                    problem=problem_text,
                    answer=float(answer),
                    operation="logarithm",
                    operands=operands,
                    difficulty=difficulty,
                    metadata={"base": operands[1]}
                ))

        return problems

    def generate_trigonometry_problems(self, count: int, difficulty: str = "easy") -> List[ArithmeticProblem]:
        """Generate trigonometry problems."""
        import math
        problems = []

        # Common angle values in radians and their results
        angles_deg = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360]
        angles_rad = [math.radians(deg) for deg in angles_deg]

        for _ in range(count):
            angle_deg = random.choice(angles_deg)
            angle_rad = math.radians(angle_deg)

            func_choice = random.choice(['sin', 'cos', 'tan'])

            if func_choice == 'sin':
                answer = round(math.sin(angle_rad), 6)
                problem_text = f"sin({angle_deg}°)"
            elif func_choice == 'cos':
                answer = round(math.cos(angle_rad), 6)
                problem_text = f"cos({angle_deg}°)"
            else:  # tan
                if angle_deg in [90, 270]:  # undefined
                    continue
                answer = round(math.tan(angle_rad), 6)
                problem_text = f"tan({angle_deg}°)"

            # Clean up common exact values
            if abs(answer - 0) < 1e-10:
                answer = 0.0
            elif abs(answer - 1) < 1e-10:
                answer = 1.0
            elif abs(answer - (-1)) < 1e-10:
                answer = -1.0
            elif abs(answer - 0.5) < 1e-10:
                answer = 0.5
            elif abs(answer - (-0.5)) < 1e-10:
                answer = -0.5
            elif abs(answer - (math.sqrt(3)/2)) < 1e-10:
                answer = round(math.sqrt(3)/2, 6)
            elif abs(answer - (math.sqrt(2)/2)) < 1e-10:
                answer = round(math.sqrt(2)/2, 6)

            problems.append(ArithmeticProblem(
                problem=problem_text,
                answer=float(answer),
                operation="trigonometry",
                operands=[float(angle_rad)],
                difficulty=difficulty,
                metadata={"angle_deg": angle_deg, "function": func_choice}
            ))

        return problems

    def generate_complex_problems(self, count: int, difficulty: str = "easy") -> List[ArithmeticProblem]:
        """Generate complex number problems."""
        import math
        problems = []

        for _ in range(count):
            # Generate Euler's identity and simple complex operations
            if random.random() < 0.5:  # Euler's identity: e^(iπ) + 1 = 0
                problem_text = "e^(i*π) + 1 ="
                answer = 0.0
                operands = [math.e, math.pi, 1]
                metadata = {"category": "euler", "source": "generated"}
            else:  # Simple complex arithmetic
                # i^2 = -1, i^3 = -i, i^4 = 1
                powers = [2, 3, 4]
                answers = [-1, 0, 1]  # Simplified for real parts
                power = random.choice(powers)
                if power == 2:
                    answer = -1.0
                    problem_text = "i^2 ="
                elif power == 3:
                    answer = 0.0  # Taking real part of -i
                    problem_text = "Re(i^3) ="
                else:  # power == 4
                    answer = 1.0
                    problem_text = "i^4 ="

                operands = [1.0, float(power)]  # Convert complex to serializable format
                metadata = {"category": "complex_power", "source": "generated"}

            problems.append(ArithmeticProblem(
                problem=problem_text,
                answer=float(answer),
                operation="complex",
                operands=[math.e, math.pi, 1] if 'euler' in problem_text else operands,
                difficulty=difficulty,
                metadata=metadata
            ))

        return problems

    def save_benchmark(self, problems: List[ArithmeticProblem], filepath: str) -> None:
        """Save benchmark problems to a JSON file."""
        import json
        
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
        
        print(f"Benchmark saved to {filepath} with {len(problems)} problems")