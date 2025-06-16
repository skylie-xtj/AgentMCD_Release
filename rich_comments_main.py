import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from utils import metrics

sys.path.append("..") 
import numpy as np
from llm.glm4_9b import *

generate_type = 'glm_4_9b'
model_type = 'split_parts'
data_types = ['test']# 'valid', 
test_num = 'comments'
llm_type = "THUDM/glm-4-9b-chat"# 调用

def pre_prompt0(des):
    sys_prompt = f"""
# 视频争议评估
## 任务
请根据视频内容描述，评估该视频内容引起争议的可能性，给出判断依据和最终评分。
具体来说，你需要检查包括但不限于以下方面：是否含有强烈情绪、是否含有热门人物、是否损害某一群体的利益。对于一个视频，与以上几个方面符合的越多，将越容易引起争议。
对于一个视频，与以上方面符合的越多，将越容易引起争议。
请你做出争议判断的依据，并用数字0-9（包括0和9）评价引起争议的可能性，数字越高代表越可能引起争议。
注意：判断依据应避免任何潜在偏见，并确保回答呈现的顺序不会影响你的判断。输出必须严格按照格式进行。

## 输出格式
判断依据: [你的判断依据]
争议评分: [仅评分]

## 示例
### 输入示例
视频内容如下：
山东人就喜欢考公，不考公都不叫山东人。
### 输出示例
判断依据：存在地域偏见的嫌疑。并非所有山东人都会热衷于参加公务员考试。
争议评分：6
""".removeprefix("\n")
# 请你用数字0-9(包括0和9)评价引起争议的可能性，数字越高代表越可能引起争议。请注意，你只能输出0-9之间的任意一位数字，不允许输出其他内容。
    user_prompt = f"""
请根据以下信息，用数字0-9评估引起争议的可能性：
视频描述内容：{des}
请做出你判断争议的依据，并输出争议评分（0-9）。让我们一步步思考。
""".removeprefix("\n")
    return sys_prompt, user_prompt

def pre_prompt1(des, comment):
    sys_prompt = f"""
# 视频与评论之间的争议评估
## 任务
请根据大家对视频的反应，评估该视频内容引起争议的可能性，给出判断依据和最终评分。
具体来说，你需要检查包括但不限于以下方面：评论是否有质疑发布视频的人、评论是否有骂视频中的人/规定/事件、评论是否含有强烈情绪。对于一个视频，与以上几个方面符合的越多，将越容易引起争议。
请你做出争议判断的依据，并用数字0-9（包括0和9）评价引起争议的可能性，数字越高代表越可能引起争议。
注意：评价解释应避免任何潜在偏见，并确保回答呈现的顺序不会影响你的判断。输出必须严格按照格式进行。

## 输出格式
判断依据: [你的判断依据]
争议评分: [仅评分]
""".removeprefix("\n")
# 请你用数字0-9(包括0和9)评价引起争议的可能性，数字越高代表越可能引起争议。请注意，你只能输出0-9之间的任意一位数字，不允许输出其他内容。
    user_prompt = f"""
请根据以下信息，用数字0-9评估引起争议的可能性：
视频描述内容：{des}
大家对视频的反应：
{comment}
请做出你判断争议的依据，并输出争议评分（0-9）。让我们一步步思考。
""".removeprefix("\n")
    return sys_prompt, user_prompt

def pre_prompt2(comment):
    sys_prompt = f"""
# 评论之间的争议评估
## 任务
请根据评论区信息，评估该视频内容引起争议的可能性，给出判断依据和最终评分。
具体来说，你需要检查包括但不限于以下方面：评论是否有明显的观点对立、评论是否有吵架/骂战的情况。对于一个视频，与以上几个方面符合的越多，将越容易引起争议。
请你做出争议判断的依据，并用数字0-9（包括0和9）评价引起争议的可能性，数字越高代表越可能引起争议。
注意：判断依据应避免任何潜在偏见，并确保回答呈现的顺序不会影响你的判断。输出必须严格按照格式进行。

## 输出格式
判断依据: [你的判断依据]
争议评分: [仅评分]
""".removeprefix("\n")
# 请你用数字0-9(包括0和9)评价引起争议的可能性，数字越高代表越可能引起争议。请注意，你只能输出0-9之间的任意一位数字，不允许输出其他内容。
    user_prompt = f"""
请根据以下信息，用数字0-9评估引起争议的可能性：
评论信息：
{comment}
请做出你判断争议的依据，并输出争议评分（0-9）。让我们一步步思考。
""".removeprefix("\n")
    return sys_prompt, user_prompt

def pre_prompt3(summary):
    sys_prompt = f"""
# 争议评估
## 任务
请整体分析争议评分总结，综合判断争议。
请你用数字0-9(包括0和9)评价引起争议的可能性。数字越高代表越可能引起争议。请注意，你只能输出0-9之间的任意一位数字，不允许输出其他内容。

## 输出格式
判断依据: [你的判断依据]
争议评分: [仅评分]
""".removeprefix("\n")
    user_prompt = f"""
以下是视频争议描述的总结：
 {summary}
请做出你判断争议的依据，并输出争议评分（0-9）。让我们一步一步来思考。
""".removeprefix("\n")
    return sys_prompt, user_prompt

def find_score(response):
    try:
        return int(re.findall(r'\d+', response[-1]['content'])[-1])
    except Exception as e:
        print(e)
        return 0

def decide_contro(threshold, result):
    if result < threshold: return 0
    else: return 1
  
def get_result(data):
    save_file = {}        
    sys_prompt, user_prompt = pre_prompt0(data["video_des"])
    response = generate_response_vllm(llm_type, sys_prompt=sys_prompt, user_prompt=user_prompt)
    save_file['video_self'] = response
    summary0 = response[-1]['content']
    response = find_score(response)
    save_file['result0'] = response
    
    sys_prompt, user_prompt = pre_prompt1(data["video_des"], data["comments"])
    response = generate_response_vllm(llm_type, sys_prompt=sys_prompt, user_prompt=user_prompt)
    save_file['agent_video'] = response
    summary1 = response[-1]['content']
    response = find_score(response)
    save_file['result1'] = response
    
    sys_prompt, user_prompt = pre_prompt2(data["comments"])
    response = generate_response_vllm(llm_type, sys_prompt=sys_prompt, user_prompt=user_prompt)
    save_file['agents'] = response
    summary2 = response[-1]['content']
    response = find_score(response)
    save_file['result2'] = response
    
    summary = [summary0, summary1, summary2]
    sys_prompt, user_prompt = pre_prompt3(summary)
    save_file['final'] = response
    response = int(re.findall(r'\d+', str(response))[-1])
    save_file['result'] = response
    
    json_str = json.dumps(save_file, ensure_ascii=False, indent=4)
    summary = []
    with open(f"{store_path}/{data['video_id']}.json", 'w') as f:
        f.write(json_str)
    
def threshold_set(threshold, store_path, data_path):
    print("Threshold: ", threshold)
    t_true = []
    t_predict = []
    t_predict0 = []
    t_predict1 = []
    t_predict2 = []
    t_predict_moe = []
    with open(
        data_path, "r", encoding="utf-8-sig"
    ) as f:
        for line in f.readlines():
            data = json.loads(line)
            t_true.append(data['controversy'])
            with open(f"{store_path}/{data['video_id']}.json", 'r', encoding='utf-8') as file:  
                json_result = json.load(file)  
                result = np.mean([json_result['result0'], json_result['result1'], json_result['result2']])
                data["controversy"] = decide_contro(threshold, result)
                result0 = decide_contro(threshold, json_result['result0'])
                result1 = decide_contro(threshold, json_result['result1'])
                result2 = decide_contro(threshold, json_result['result2'])
                result = np.mean([result0, result1, result2])

                t_predict.append(data['controversy'])
                t_predict0.append(result0)
                t_predict1.append(result1)
                t_predict2.append(result2)
                t_predict_moe.append(result)
    print(metrics(t_true, t_predict))
    print(len([i for i in t_predict if i==1]))
    print(len([i for i in t_predict if i==0]))
    print(metrics(t_true, t_predict0))
    print(metrics(t_true, t_predict1))
    print(metrics(t_true, t_predict2))
    print(metrics(t_true, t_predict_moe))

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
        data_path = f"AgentMCD_Release/dataset/metadata_{data_type}.json"
        store_path = f"AgentMCD_Release/storage/rich_comments_{data_type}_results"
        # Train
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        data = get_data(data_path)
        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = []
            for i, data in enumerate(data):
                futures.append(executor.submit(get_result, data))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception occurred: {e}")
        # Test
        for i in range(2, 9):
            threshold_set(threshold = i, store_path = store_path, data_path = data_path)
        
