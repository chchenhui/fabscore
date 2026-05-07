from openai import OpenAI


# Prompt to add when generating a section
prompt = """
Cite any relevant sources from Semantic Scholar.
Use the format \\cite{full original title of the paper__year of the paper}.
Do not leave non trivial statements without citations. 
Do not reference non existing papers. 
There is no need to generate the bibliography section.

Thank you!
"""


demo_prompt = f"""
Write the related work section of a research paper on claim verification.
Produce the result as a single LaTeX file. 
Be concise.

{prompt}
""".strip()


client = OpenAI(api_key="key-here")

response = client.responses.create(
    model = "gpt-5-nano",
    instructions = "You are a computer science researcher that has to write a new research paper.",
    input = demo_prompt,
)

with open("section.tex", "w") as f:
    f.write(response.output_text)