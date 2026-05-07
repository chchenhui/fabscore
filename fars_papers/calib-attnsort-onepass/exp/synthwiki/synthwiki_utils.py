import datetime
import hashlib
import os
import random
import numpy as np

def log(message):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"[{current_time}] {message}"
    print(full_message, flush=True)

def hash_string(input_string, len=8):
    # Encode the input string to bytes, then hash it with MD5
    md5_hash = hashlib.md5(input_string.encode())
    # Convert the hash to a hexadecimal string
    hex_hash = md5_hash.hexdigest()
    # Return the first 8 characters
    return hex_hash[:len]

def check_file_exists(filename):
    return os.path.isfile(filename)

def genJunkContext(contexts, limit, tokenizer):
    random.shuffle(contexts)
    token_count = 0
    sublist = []
    for paragraph in contexts:
        paragraph_tokens = tokenizer.encode(paragraph)
        paragraph_token_count = len(paragraph_tokens)
        if token_count + paragraph_token_count > limit:
            break
        else:
            token_count += paragraph_token_count
            sublist.append(paragraph)
    return sublist

def insertIntoJunk(junk, doc, insert_place):
    if insert_place == 'halfway':
        pos_to_insert = int(np.floor(len(junk)/2))  # type: ignore
    elif insert_place == 'random':
        pos_to_insert = np.random.randint(len(junk))
    elif insert_place == 'first':
        pos_to_insert = 0
    elif insert_place == 'last':
        pos_to_insert = len(junk)-1
    else:
        raise RuntimeError(insert_place)

    junk[pos_to_insert] = doc

    return junk, pos_to_insert


def checkCorrectness(answer, true_answer):
    answer_list = true_answer.split(' | ')

    answer = answer.replace('\n', '')
    answer = answer.replace('’', '')

    answer = answer.strip().strip('"')
    answer = answer.strip().strip("'")
    answer = answer.replace("'", '')

    answer_list = [t.strip().strip('"') for t in answer_list]
    answer_list = [t.strip().strip("'") for t in answer_list]
    answer_list = [t.replace("'", '') for t in answer_list]
    answer_list = [t.replace("’", '') for t in answer_list]

    correct = 0
    for true_answer in answer_list:
        if true_answer.lower() in answer.lower():
            correct = 1

    return correct