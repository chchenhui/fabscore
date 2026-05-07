import time
from openai import OpenAI
from tqdm import tqdm
import threading

class APIModel:
    """OpenAI client-based model for AutoSurvey - compatible with vLLM"""
    
    def __init__(self, model, api_key, api_url) -> None:
        # For vLLM, use base_url without the /chat/completions part
        if api_url.endswith('/chat/completions'):
            base_url = api_url.replace('/chat/completions', '')
        elif api_url.endswith('/v1'):
            base_url = api_url
        else:
            base_url = api_url.rstrip('/') + '/v1'
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key if api_key else "EMPTY"  # vLLM doesn't need real key
        )
        self.model = model
        
    def __req(self, text, temperature, max_try=5):
        for attempt in range(max_try):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": text}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_try - 1:
                    print(f"Error after {max_try} attempts: {e}")
                    return None
                time.sleep(0.2)
        return None
    
    def chat(self, text, temperature=1):
        response = self.__req(text, temperature=temperature, max_try=5)
        return response

    def __chat(self, text, temperature, res_l, idx):
        response = self.__req(text, temperature=temperature)
        res_l[idx] = response
        return response
        
    def batch_chat(self, text_batch, temperature=0):
        max_threads = 15  # limit max concurrent threads using model API
        res_l = ['No response'] * len(text_batch)
        thread_l = []
        for i, text in zip(range(len(text_batch)), text_batch):
            thread = threading.Thread(target=self.__chat, args=(text, temperature, res_l, i))
            thread_l.append(thread)
            thread.start()
            while len(thread_l) >= max_threads: 
                for t in thread_l:
                    if not t.is_alive():
                        thread_l.remove(t)
                time.sleep(0.3)  # Short delay to avoid busy-waiting

        for thread in tqdm(thread_l):
            thread.join()
        return res_l