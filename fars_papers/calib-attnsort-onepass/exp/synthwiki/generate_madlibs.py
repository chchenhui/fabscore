import numpy as np
import pandas as pd
import argparse
import openai 
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

parser = argparse.ArgumentParser(description='Args')
parser.add_argument('--save_here', 
                    default=f'{__file__}/data/madlibs/madlibs1.csv',
                    help='Where save?')
parser.add_argument('--num_people', 
                    default=500,
                    type=int,
                    help='Where results to save?')
parser.add_argument('--openai_key', 
                    help='Your OpenAI key')
args = parser.parse_args()

openai.api_key = args.openai_key

def askGPT(prompt, model='gpt-4'):
    ## JUST KEEP TRYING!
    retry_limit = 10
    retry_count = 0

    while retry_count < retry_limit:
        try:
            response = openai.ChatCompletion.create(
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant bot.'},
                    {'role': 'user', 'content': prompt},
                ],
                model=model
                )
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
            retry_count += 1
            # Probably hit rate limit, so let's just wait.
            time.sleep(60)

    if retry_count == retry_limit:
        print("Reached maximum retry limit.")

    reply = response['choices'][0]['message']['content']

    return reply

print("GETTING ORIGINS")
origins_prompt = "Give me a list of 50 origins for people (example: Canadian, Texan, European, Monagasque), output it as a comma separated list. Output the list only, nothing else."

duped_origins = []
for k in range(15):
    print(k)
    tmp = askGPT(origins_prompt, model='gpt-3.5-turbo')
    print(tmp)
    duped_origins = duped_origins + tmp.split(", ")
    
origins = np.unique(np.array(duped_origins).flatten())
# Hack to make sure it's just the origin
origins = [s for s in origins if len(s) < 30]
print(len(origins))

print("GETTING JOBS")
jobs_prompt = "Give me a list of 50 jobs that people can become famous for (example: programmer, entrepreneur, taxi driver, basketball player), output it as a comma separated list. Output the list only, nothing else."

duped_jobs = []
for k in range(15):
    print(k)
    tmp = askGPT(jobs_prompt, model='gpt-3.5-turbo')
    print(tmp)
    duped_jobs = duped_jobs + tmp.split(", ")
    
jobs = np.unique(np.array(duped_jobs).flatten())
jobs = [s for s in jobs if len(s) < 40]
print(len(jobs))

print("Geting names")
non_duped_example_jobs = []
non_duped_example_orgins = []
non_duped_example_names = []
for j in range(args.num_people):
    an_origin = random.choice(origins)
    a_job = random.choice(jobs)
    name_prompt = f"I am writing a novel, help me come up with a name for a famous {an_origin} {a_job}. Do not output the name of someone who already exists. Output only the name, no explanation."
    name = askGPT(name_prompt, model='gpt-3.5-turbo')
    # Hack to make sure it's just the name.
    if len(name) < 30:
        print(name)
        if name not in non_duped_example_names:
            non_duped_example_names.append(name)
            non_duped_example_orgins.append(an_origin)
            non_duped_example_jobs.append(a_job)
        
questions = [
    "In which city was PERSON born?",
    "What year was PERSON born?",
    "Where did PERSON go to college?",
    "What is the name of PERSON's spouse?",
    "What is the name of the first company PERSON worked at?",
    "What is the company PERSON founded called?",
    "What is the title of the film PERSON directed?",
    "Who is PERSON's idol?",
    "What is the name of PERSON's pet?",
    "What is PERSON's favorite color?",
    "Where did PERSON go to high school?",
    "What is the name of PERSON's best friend?",
    "What is the title of PERSON's favorite movie?",
    "In what year did PERSON get married?",
    "What is the title of PERSON's favorite book?",
    "What is the name of PERSON's first child?",
    "What is the name of PERSON's favorite sports team?",
    "In which country was PERSON born?",
    "What was the title of PERSON's PhD thesis?",
    "What sport does PERSON play?"
]

madlibs = pd.DataFrame({})

def doOne(name, job, origin):
    q = random.sample(questions, 2)
    question1 = q[0].replace("PERSON", name)
    question2 = q[1].replace("PERSON", name)
    wiki_prompt = f"""Please write a one paragraph wikipedia article for a famous {origin} {job} named {name}.
    Make sure the article contains information that can answer the following questions:
    {question1}
    {question2}
    Output the article only, no extraneous explanation.
    """
    wiki = askGPT(wiki_prompt)
    print("WIKI EXAMPLE")
    print(wiki)
    q1_prompt = f"""Here is a short wikipedia article. 
    ##ARTICLE
    {wiki}

    Can you answer the following question?

    ##QUESTION
    {question1}

    Keep the answer as short as possible. If you can answer in one or two words, do that.
    """
    a1 = askGPT(q1_prompt)

    q2_prompt = f"""Here is a short wikipedia article. 
    ##ARTICLE
    {wiki}

    Can you answer the following question?

    ##QUESTION
    {question2}

    Keep the answer as short as possible. If you can answer in one or two words, do that.
    """
    a2 = askGPT(q2_prompt)

    row1 = pd.DataFrame({'context': wiki, 
                         'question': question1,
                         'answer': a1
                        }, index=[0])
    row2 = pd.DataFrame({'context': wiki, 
                         'question': question2,
                         'answer': a2
                        }, index=[0])
    row = pd.concat([row1, row2], ignore_index=True)
    return row

def worker(n, j, o):
    return doOne(name=n, job=j, origin=o)

data = list(zip(non_duped_example_names, non_duped_example_jobs, non_duped_example_orgins))

pbar = tqdm(total=len(data), desc='Processing data')

results = []

print("RUNNING WHOLE THING")

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(worker, n, j, o): (n, j, o) for (n, j, o) in data}
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        pbar.update(1)

pbar.close()

madlibs = pd.concat(results, ignore_index=True)
madlibs.to_csv(args.save_here, index=False)
