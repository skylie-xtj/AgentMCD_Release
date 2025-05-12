import json
import pickle
import re
import warnings
import os
from tqdm import tqdm
from llm.glm4_9b import *
warnings.filterwarnings("ignore")
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append("AgentMCD_Release") 

int_agents = 10
def get_data(data_path='AgentMCD_Release/dataset/metadata_test.json'):
    """
    Input:data_path: path to data file
    Output:result:list of dict : {'comments':"*********",'video_des':"*************"}
    用来获取用于测试的数据
    """
    result = []
    with open(data_path, "r", encoding="utf-8-sig") as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            temp = {}
            data = json.loads(line)
            comments = ""
            for i in data["comments"].keys():
                comments += re.sub(
                    r"@\w+\s?", "", data["comments"][i]["comments"]
                )  # comment_content
            comments = comments.replace("\r", " ")
            temp["comments"] = comments
            temp['video_des'] = data['video_des']
            temp['video_id'] = data['video_id']
            result.append(temp)
    return result

def prepare_agent_role_prompt(des):
    sys_prompt = f"""请你仔细阅读视频内容和相关的评论信息，总结观看视频的用户类型，并提取每类用户的核心观点。
每类用户输出严格按照如下格式：
用户类型: [这里只写用户类型]
核心观点: [这里只写核心观点]
"""
    user_prompt = f"""视频内容如下：
{des}
"""
    return sys_prompt, user_prompt

save_file = {}
user_role_details = {}
def generate_role(data):
    user_role = ''
    pattern1 = re.compile(r"用户类型:(.+)|用户类型\d+:(.+)")
    pattern2 = re.compile(r"核心观点:(.+)|核心观点\d+:(.+)", flags=re.DOTALL)
    user_role_detail = {}
    if data['video_id'] in user_role_details.keys(): return   
    sys_prompt, user_prompt = prepare_agent_role_prompt(data['video_des'].strip('\n'))
    user_role = generate_response_vllm('THUDM/glm-4-9b-chat', sys_prompt=sys_prompt, user_prompt=user_prompt)
    save_file[data['video_id']] = user_role
    user_roles = user_role[-1]['content'].strip('\n').split('\n\n')
    for j in range(len(user_roles)):
        role = re.findall(pattern1, user_roles[j])[0][0].replace(' ', '').replace('\n', '')
        des = re.findall(pattern2, user_roles[j])[0][0].replace(' ', '').replace('\n', '')
        user_role_detail[role] = des
        user_role_details[data['video_id']] = user_role_detail

def multi_thread_run(datas):
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i, data in enumerate(datas):
            futures.append(executor.submit(generate_role, data))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred: {e}")

def change_json_pkl():
    with open(log_path, 'r', encoding='utf-8') as file:  
        save_file = json.load(file)  
    pattern1 = re.compile(r"用户类型:(.+)|用户类型\d+:(.+)")
    pattern2 = re.compile(r"核心观点:(.+)|核心观点\d+:(.+)", flags=re.DOTALL)
    for data in datas:
        user_role = save_file[data['video_id']]
        user_roles = user_role[-1]['content'].strip('\n').split('\n\n')
        user_role_detail = {}
        for j in range(len(user_roles)):
            try:
                role = re.findall(pattern1, user_roles[j])[0][0].replace(' ', '').replace('\n', '')
                des = re.findall(pattern2, user_roles[j])[0][0].replace(' ', '').replace('\n', '')
                user_role_detail[role] = des
                user_role_details[data['video_id']] = user_role_detail
            except:
                continue
    with open(output_path, "wb") as f:
        pickle.dump(user_role_details, f)
        
if __name__ == "__main__":
    datas = get_data(data_path='AgentMCD_Release/dataset/metadata_test.json')
    if not os.path.exists(f'AgentMCD_Release/dataset/agent_role'):
        os.mkdir(f'AgentMCD_Release/dataset/agent_role')
    log_path = f'AgentMCD_Release/dataset/agent_role.json'
    output_path = f"AgentMCD_Release/dataset/agent_role.pkl"
    # multi_thread_run(datas) 
    # json_str = json.dumps(save_file, ensure_ascii=False, indent=4)
    # with open(log_path, 'w', encoding='utf-8') as f:
    #     f.write(json_str)
    # with open(output_path, "wb") as f:
    #     pickle.dump(user_role_details, f)
    change_json_pkl()
