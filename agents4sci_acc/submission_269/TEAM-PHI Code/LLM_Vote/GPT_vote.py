import os
from openai import AzureOpenAI
import json

endpoint = ""

# model_name = "gpt-35-turbo"
# deployment = "gpt-35-turbo"

model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

subscription_key = ""
api_version = ""

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


# guideline_path = "vote_guideline_info.txt"
guideline_path = "vote_guideline_independent.txt"


with open(guideline_path, "r", encoding="utf-8") as f:
    guideline = f.read()

llm_results = """

"""

response = client.chat.completions.create(
    model=deployment,
    messages=[
        {
            "role": "system",
            "content": "You are an experienced evaluation agent who helps with evaluate results of several de-identification (deid) agents.",
        },
        {
            "role": "user",
            "content": guideline + "\n" + llm_results,
        },
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
)

output_text = response.choices[0].message.content

print(output_text)
