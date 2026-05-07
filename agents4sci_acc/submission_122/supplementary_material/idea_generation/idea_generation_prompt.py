  
def build_zero_shot_prompt(input, num_ideas):
    prompt = f"""
        Your task is to generate a list of original, feasible, and impactful research ideas.
        The goal is to design research projects that focus on enhancing table grounding techniques or introducing reasoning steps that are useful for evaluation and performance improvement.
        The information below includes details about the dataset, related work, and parts of the introduction: {input}  
        Return the output **only in valid JSON**, without extra commentary or explanations.
        Follow this exact JSON schema:
        {{
        "ideas": [
            {{
            "title": "string - concise and catchy research title",
            "core_idea": "string - 2–3 sentence summary of the research",
            "hypothesis": "string - main hypothesis to be tested",
            "why_it_matters": "string - significance and potential impact of pursuing this research",
            "possible_methods": ["list of suggested experimental designs"],
            "experimental_design": ["guideline details on how to generate the code and how to run experiments]
            }}
        ]
        }}
        Generate {num_ideas} research ideas in this format.
		"""
    return prompt