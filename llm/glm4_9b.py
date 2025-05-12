import time
import requests
from openai import OpenAI
import random

# vllm
client1 = OpenAI(base_url=f'http://localhost:8880/v1/', api_key='sk-xxx')
# client2 = OpenAI(base_url=f'http://localhost:8882/v1/', api_key='sk-xxx')
# client3 = OpenAI(base_url=f'http://localhost:8883/v1/', api_key='sk-xxx')
# client4 = OpenAI(base_url=f'http://localhost:8885/v1/', api_key='sk-xxx')
# client5 = OpenAI(base_url=f'http://localhost:8885/v1/', api_key='sk-xxx')
# client7 = OpenAI(base_url=f'http://localhost:8887/v1/', api_key='sk-xxx')
# clients = [client1, client2, client3, client4]
clients = [client1]

def generate_response_vllm(llm_type='glm-4-9b-chat', temperature=0, user_prompt=None, sys_prompt=None):
    llm_type='glm4_9b'
    response = None
    cnt = 1
    start_time = time.time()  # Record the start time
    max_tokens = 30000
    time_limit = 120
    while response is None:
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:  # If time exceeds limit (60 seconds)
            print("Time limit exceeded, adjusting max_tokens.")
            max_tokens = 2048 
        client = clients[random.randint(0, len(clients) - 1)]
        try:
            if sys_prompt is not None:
                messages = [ 
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                response = client.chat.completions.create(
                    model=llm_type,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=time_limit,
                    extra_body={
                        "stop_token_ids": [151329, 151336, 151338]
                    }
                )
            else:
                messages = [
                    {"role": "user", "content": user_prompt}
                ]
                response = client.chat.completions.create(
                    model=llm_type,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={
                        "stop_token_ids": [151329, 151336, 151338]
                    }
                )
        except Exception as e:
            print(e)
            cnt += 1
            if cnt >= 10:
                max_tokens = 2048
                print("Time limit exceeded, adjusting max_tokens.")
    response = response.choices[0].message.content.strip()  # ['content']
    messages.append({"role": "assistant", "content": response})
    return messages