import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append("Agent_mcd") 
import requests
from llm.glm4_9b import generate_response
from llm.secret_file import api_keys
from tqdm import tqdm
from utils import metrics

data_types = ['valid', 'test']
test_num = 'cot_0'# wo_comment_
model_type = 'glm4_9b'# 保存
llm_type = "THUDM/glm-4-9b-chat"# 调用

def get_result(data, api_key, store_path):
    save_file = {}        
    sys_prompt, user_prompt = pre_prompt1(data["video_des"], data["comments"])
    response = generate_response(llm_type, sys_prompt=sys_prompt, user_prompt=user_prompt, api_key=api_key)
    save_file['log'] = response
    response = int(re.findall(r'\d+', response[-1]['content'])[-1])
    save_file['result'] = response
    json_str = json.dumps(save_file, ensure_ascii=False, indent=4)
    with open(f"{store_path}/{data['video_id']}.json", 'w') as f:
        f.write(json_str)
    
def threshold_set(threshold, store_path, data_path):
    print("Threshold: ", threshold)
    t_true = []
    t_predict = []
    with open(
        data_path, "r", encoding="utf-8-sig"
    ) as f:
        t_ture_dict = {}
        for line in f.readlines():
            data = json.loads(line)
            t_ture_dict[data['video_id']] = data['controversy']
    for i in os.listdir(store_path):
        with open(f"{store_path}/{i}", 'r', encoding='utf-8-sig') as file:  
            data = json.load(file)  
            if data['result'] < threshold:
                data["controversy"] = 0
            else:
                data["controversy"] = 1
            t_predict.append(data['controversy'])
            t_true.append(t_ture_dict[i.replace('.json', '')])
            
    results = metrics(t_true, t_predict)
    print(results)
    print(len([i for i in t_predict if i==1]))
    print(len([i for i in t_predict if i==0]))


def pre_prompt1(des, comment):
    sys_prompt = f"""
# 争议评估
## 任务
请你根据与视频和评论相关的信息，评估内容引起争议的可能性。
具体而言，你需要检查以下三个方面：
1. 视频本身是否容易存在争议：是否含有强烈情绪、是否含有热门人物、是否损害某一群体的利益
2. 视频和评论之间是否存在争议：评论是否有质疑发布视频的人、评论是否有骂视频中的人/规定/事件、评论是否含有强烈情绪
3. 评论和评论之间是否存在争议：评论是否有明显的观点对立、评论是否有吵架/骂战的情况
对于一个视频，与以上三个方面符合的越多，将越容易引起争议。
请你做出争议判断的依据，并用数字0-9（包括0和9）评价引起争议的可能性，数字越高代表越可能引起争议。
注意：判断依据应避免任何潜在偏见，并确保回答呈现的顺序不会影响你的判断。输出必须严格按照格式进行。

## 输出格式
判断依据: [你的判断依据]
争议评分: [仅评分]
""".removeprefix("\n")
# 请你用数字0-9(包括0和9)评价引起争议的可能性，数字越高代表越可能引起争议。请注意，你只能输出0-9之间的任意一位数字，不允许输出其他内容。
    user_prompt = f"""
请根据以下信息，用数字0-9评估引起争议的可能性：
视频描述内容：{des}
评论信息：
{comment}
请做出你判断争议的依据，并输出争议评分（0-9）。
""".removeprefix("\n")
    return sys_prompt, user_prompt

def pre_prompt0(des, comment):
    sys_prompt = f"""
# 争议评估
## 任务
请你根据与视频和评论相关的信息，评估内容引起争议的可能性。
具体而言，你需要检查视频本身是否容易存在争议：是否含有强烈情绪、是否含有热门人物、是否损害某一群体的利益
对于一个视频，与以上符合的越多，将越容易引起争议。
请你做出争议判断的依据，并用数字0-9（包括0和9）评价引起争议的可能性，数字越高代表越可能引起争议。
注意：判断依据应避免任何潜在偏见，并确保回答呈现的顺序不会影响你的判断。输出必须严格按照格式进行。

## 输出格式
判断依据: [你的判断依据]
争议评分: [仅评分]
""".removeprefix("\n")
# 请你用数字0-9(包括0和9)评价引起争议的可能性，数字越高代表越可能引起争议。请注意，你只能输出0-9之间的任意一位数字，不允许输出其他内容。
    user_prompt = f"""
请根据以下信息，用数字0-9评估引起争议的可能性：
视频描述内容：{des}
请做出你判断争议的依据，并输出争议评分（0-9）。
""".removeprefix("\n")
    return sys_prompt, user_prompt

def get_data(data_path):
    """
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

if __name__ == "__main__":
    for data_type in data_types:
        data_path = f"Agent_mcd/dataset/metadata_{data_type}.json"
        store_path = f"Agent_mcd/storage/baselines/{model_type}_{data_type}_{test_num}"
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        data = get_data(data_path)
        with ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
            futures = []
            for i, data in enumerate(data):
                futures.append(executor.submit(get_result, data, api_keys[i % len(api_keys)], store_path))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception occurred: {e}")
        for i in range(2, 9):
            threshold_set(threshold = i, store_path = store_path, data_path = data_path)
