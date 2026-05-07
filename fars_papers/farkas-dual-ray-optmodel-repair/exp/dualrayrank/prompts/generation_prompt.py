"""Attempt-0 prompt template for LP generation from natural-language optimization problems.

Adapted from MAMO's few-shot LP prompt (prompt1 in combined_prompts.json) with
added constraint-naming convention (c0001, c0002, ...) required for downstream
repair feedback based on constraint names.
"""

_SYSTEM_PROMPT = (
    "You are an expert in mathematical optimization. You translate natural language "
    "optimization problems into .lp format files that can be solved by LP/MIP solvers."
)

_FEW_SHOT_TEMPLATE = """\
Assume you are a virtual assistant with expertise in optimization, specifically in creating .lp files for linear programming problems. Your task is to translate given natural language problems into optimization models, formatted as .lp files.
When you receive a question, it might include mathematical expressions in LaTeX format. Your job is to interpret these expressions accurately and model the problem in .lp format. Your response must adhere to the following guidelines:
- The optimization model must be written in .lp format, adhering to the conventions and syntax appropriate for linear programming problems.
- The model should be designed so that solving it yields the optimal value, which directly answers the question posed.
- Your response should be an entire .lp file, ready to be processed by a linear programming solver. Ensure that the file contains no comments or extraneous content beyond the model itself.
- Handle LaTeX expressions with care to ensure that the mathematical aspects of the problem are accurately represented in the .lp model.
- If the solution needs to be rounded to an integer, make use of the 'General' integer constraint in the .lp file to specify integer variables.
- Name every constraint sequentially as c0001, c0002, c0003, ... (with the "name:" prefix in the Subject To section).
- Include explicit bounds for all variables in the Bounds section.
Here comes the examples:
[Example_1]
(the input)
A manufacturing company produces two types of products: $X$ and $Y$. The production cost for each unit of product $X$ is $\\$50$, while the cost for each unit of product $Y$ is $\\$10$. There are constraints in the production process, such that twice the number of units produced from product $X$, plus the number of units from product $Y$, cannot exceed 200 due to resource limitations. In addition, to meet market demand, four times the number of units produced from product $X$, plus the number of units from product $Y$, must be at least 50.
Considering these constraints and given that both products can only be produced in whole numbers due to their physical nature, what is the minimum total cost needed for producing these items while satisfying all conditions? Provide your answer rounded to the nearest dollar.
Your response:
Minimize
obj: 50 x + 10 y
Subject To
c0001: 2 x + y <= 200
c0002: 4 x + y >= 50

Bounds
x >= 0
y >= 0

Generals
x
y

End
Please craft the .lp file according to these instructions, focusing on delivering a model that is directly solvable to obtain the answer.
And Please follow the syntax like examples to write the .lp file.
Here comes the question:
{question}
Generate the contents of an .lp file for this problem, starting with the objective function and followed by the constraints, without any additional sentences. The constraints should be formatted as 'variable + variable >= number' for inequalities, all the variables should be on the left hand side of the inequality. Ensure there is a space between variables and their coefficients. Name every constraint sequentially as c0001, c0002, c0003, etc.
Your response:
"""


def build_generation_prompt(question: str) -> str:
    """Build the attempt-0 generation prompt for a given optimization problem."""
    return _FEW_SHOT_TEMPLATE.format(question=question)


def get_system_prompt() -> str:
    return _SYSTEM_PROMPT
