import torch
import numpy as np
import pandas as pd
from synthwiki_utils import log, checkCorrectness



def generateAttnPrompt(docs, question, tokenizer, prompt_type='wizard'):
    # Add newlines for docs
    docs = ["DOCUMENT: " + string + "\n\n" for string in docs]
    
    base_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction\nHere is some information you will use to answer a question. Some of the information may be irrelevant.\n\n### Information\n"""
    if prompt_type == 'together_instruct':
        print("USING 2GETHER TYPE PROMPT")
        base_prompt = """[INST]\nHere is some information you will use to answer a question. Some of the information may be irrelevant.\n\n### Information\n"""

    current_prompt = tokenizer(base_prompt, return_tensors='pt', return_token_type_ids=False)
    docs_tokenized = []
    for d in docs:
        docs_tokenized.append(tokenizer(d, return_tensors='pt', return_token_type_ids=False, add_special_tokens = False))

    docs_idx_range = []

    for d in docs_tokenized:
        d_start = len(current_prompt['input_ids'][0])
        new_prompt = current_prompt.copy()
        new_prompt['input_ids'] = torch.cat([current_prompt['input_ids'], d['input_ids']], axis=1)
        new_prompt['attention_mask'] = torch.cat([current_prompt['attention_mask'], d['attention_mask']], axis=1)
        d_end = len(new_prompt['input_ids'][0])
        docs_idx_range.append([d_start, d_end])
        current_prompt = new_prompt

    end_prompt = f"""\n\n### Question\n{question}\n\nPlease return only the answer to the question. Answer concisely.\n\n### Answer\n"""
    if prompt_type == 'together_instruct':
       end_prompt = f"""\n\n### Question\n{question}\n\nPlease return only the answer to the question. Answer concisely.\n[/INST]\n\n"""
     
    end_prompt_tokenized = tokenizer(end_prompt, return_tensors='pt', return_token_type_ids=False, add_special_tokens = False)
    final_prompt = current_prompt.copy()
    final_prompt['input_ids'] = torch.cat([current_prompt['input_ids'], end_prompt_tokenized['input_ids']], axis=1)
    final_prompt['attention_mask'] = torch.cat([current_prompt['attention_mask'], end_prompt_tokenized['attention_mask']], axis=1)
    final_prompt_len = len(final_prompt['input_ids'][0])
    return final_prompt, docs_idx_range, final_prompt_len


def getCitation(attentions, doc_idx_range):
    # Downsample output str
    citations = pd.DataFrame({})
    # print(f"attentions: {len(attentions)} {attentions}")
    att_example = attentions[0]
    n_layers = len(att_example)
    n_heads = att_example[0].shape[1]
    data_list = []
    log("Starting big for loop")
    for gen_token in np.arange(0, len(attentions)):
        attn = attentions[gen_token]
        for d in range(len(doc_idx_range)):
            for which_layer in range(n_layers):
                layer_attn = attn[which_layer].squeeze()
                attn_sum = torch.sum(layer_attn[:, doc_idx_range[d][0]:doc_idx_range[d][1]], dim=-1).cpu().data.float().numpy()
                for head in range(n_heads):
                    data_list.append({
                        'generation_token': gen_token,
                        'doc': d,
                        'layer': which_layer,
                        'head': head,
                        'attn_sum': attn_sum[head]
                    })
    citations = pd.DataFrame(data_list)
    citations = citations.groupby(['generation_token', 'doc', 'layer'])['attn_sum'].mean().reset_index()
    log("Done big for loop")

    return citations


def askLLMWithCitation(docs, question, model, tokenizer, prompt_type='wizard'):
    tokenized_prompt, doc_idx_range, final_prompt_len = generateAttnPrompt(docs=docs, 
            question=question, 
            tokenizer=tokenizer,
            prompt_type=prompt_type)

    tokenized_prompt.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **tokenized_prompt, max_new_tokens=36, 
            output_attentions=True, return_dict_in_generate=True
        )

    output_str = tokenizer.decode(outputs['sequences'][0][final_prompt_len:])
    citations = getCitation(outputs['attentions'], doc_idx_range)

    return output_str, citations





def askLLMTopp(*, docs, question, answer, tokenizer, model, attn_topp=None, attn_topk=None):
    tokenized_prompt, doc_idx_range, final_prompt_len = generateAttnPrompt(docs=docs, 
        question=question, 
        tokenizer=tokenizer)
            
    tokenized_prompt.to(model.device)
    with torch.no_grad():
        output = model.generate(**tokenized_prompt, max_new_tokens=36, attn_topp=attn_topp, attn_topk=attn_topk).cpu()
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer_text = tokenizer.decode(output[0][final_prompt_len:], skip_special_tokens=True)
    correct = checkCorrectness(answer_text, answer)

    return correct, answer_text, output_text


def getAttnScores(prompt, model, doc_idx_range, max_new_tokens=1):
    log("getAttnScores start")
    # This is just used to get token 0 attentions
    with torch.no_grad():
        prompt.to(model.device)
        outputs = model.generate(
            **prompt, max_new_tokens=max_new_tokens, 
            output_attentions=True, return_dict_in_generate=True,
        )
    attentions = outputs['attentions']
    citations = getCitation(attentions, doc_idx_range)

    #filtered_citations = citations[citations['layer'].isin([10, 30])]
    filtered_citations = citations[citations['generation_token'] == 0]
    log("getAttnScores end")

    return filtered_citations.groupby('doc')['attn_sum'].mean().values, citations


def evalModelOnQuestion(docs, question, tokenizer, model, true_answer, prompt_type='wizard'):

    prompt, _, prompt_len = generateAttnPrompt(docs=docs, 
        question=question, 
        tokenizer=tokenizer,
        prompt_type=prompt_type)
    
    prompt.to(model.device)
    with torch.no_grad():
        sorted_outputs = model.generate(
            **prompt, max_new_tokens=36, 
            output_attentions=False, return_dict_in_generate=False
        )
        
    model_answer = tokenizer.decode(sorted_outputs[0][prompt_len:])
    correct = checkCorrectness(model_answer, true_answer)

    return model_answer, correct

def attnSortLLM(docs, question, true_answer, model, tokenizer, topk=[5, 10, 20, 50, 1000], best_doc_last=False, num_iters=1, true_doc_pos=0, prompt_type='wizard'):
    log(f"Starting re-sorting...")
    rows = []
    N = len(docs)
    vanilla_answer, vanilla_correct = evalModelOnQuestion(
        docs=docs, 
        question=question, 
        tokenizer=tokenizer, 
        model=model, 
        true_answer=true_answer,
        prompt_type=prompt_type
    )
    
    rows.append((vanilla_answer, vanilla_correct, 0, -1, true_doc_pos))
    log(f"Original true doc pos: {true_doc_pos} (total: {len(docs)})")
    citations = None
    for iter_idx in range(1, num_iters + 1):
        prompt, doc_idx_range, prompt_len = generateAttnPrompt(
            docs=docs, 
            question=question, 
            tokenizer=tokenizer,
            prompt_type=prompt_type
        )
        attn_scores, citations = getAttnScores(prompt, model, doc_idx_range)

        score_order = np.argsort(-attn_scores)  # type: ignore
        true_doc_pos = np.where(score_order == true_doc_pos)[0][0]
        docs = [docs[i] for i in score_order]
        if best_doc_last:
            docs = docs[::-1]
            true_doc_pos = (len(docs) - 1 - true_doc_pos)
        
        log(f"True doc pos at iter {iter_idx}: {true_doc_pos} ( {len(docs)} total )")

        for k in topk:
            log(f"k={k}")
            if k > 0:
                topk_docs = docs[-k:] if best_doc_last else docs[:k]
            else:
                topk_docs = docs
            
            sorted_answer, sorted_correct = evalModelOnQuestion(
                docs=topk_docs, 
                question=question, 
                tokenizer=tokenizer, 
                model=model, 
                true_answer=true_answer,
                prompt_type=prompt_type
            )
            rows.append((sorted_answer, sorted_correct, iter_idx, k, true_doc_pos))
    
    
    row = pd.DataFrame({
        'question': question,
        'true_answer': true_answer,
        'answers': [r[0] for r in rows],
        'correct': [r[1] for r in rows],
        'sort_iters': [r[2] for r in rows],
        'topk': [r[3] for r in rows],
        'doc_pos_sorted': [r[4] for r in rows],
        'doc_rank_sorted': [((len(docs) - r[4] - 1) if best_doc_last else r[4]) for r in rows],
    })
    log("Done")
    return row, citations, docs