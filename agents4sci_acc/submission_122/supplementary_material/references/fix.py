import os
import argparse
import json
from openai import OpenAI



def fix_citations(
    section: str, 
    openai_client, 
    model_card = "gpt-5-nano"
) -> str:
    prompt = f"""
    {section}

    Given this section of a research paper, convert all citations into the format \\cite{{full original title of the paper__year of the paper}}.
    Output the same section with the addition of the citations.
    Do not make modifications to the content.
    Cite any relevant sources from Semantic Scholar.
    Do not reference non existing papers. 
    There is no need to generate the bibliography section.

    Thank you!
    """.strip()

    response = openai_client.responses.create(
        model = model_card,
        instructions = "You are a computer science researcher that has to write a new research paper.",
        input = prompt,
    )

    return response.output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Citations correction.")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model to use")
    parser.add_argument("--section-tex", type=str, required=True, help="Source tex file of the section")
    parser.add_argument("--out-path", type=str, default="./section-cited.tex", help="Output file where the section with references is saved")
    openai_group = parser.add_mutually_exclusive_group(required=True)
    openai_group.add_argument("--openai-key", type=str, help="Plain OpenAI key")
    openai_group.add_argument("--openai-key-path", type=str, help="Path to file containing the OpenAI key")
    openai_group.add_argument("--openai-key-env", type=str, help="Name of environment variable containing the OpenAI key")
    args = parser.parse_args()

    # Prepare OpenAI client
    if args.openai_key is not None:
        openai_key = args.openai_key
    elif args.openai_key_path is not None:
        openai_key = open(args.openai_key_path, "r").read()
    else:
        openai_key = os.environ[args.openai_key_env]
    client = OpenAI(api_key=openai_key)

    # Load section
    with open(args.section_tex, "r") as f:
        section = f.read()

    new_section = fix_citations(section, client, args.model)

    with open(args.out_path, "w") as f:
        f.write(new_section)