import json
import os
import random
import re
from scipy import stats
from utils import metrics

random.seed(42)

def json_cal_result(test_type, threshold = 5, save = False):
    print("Threshold: ", threshold)
    true_label = []
    predict_label = []
    predict_video_label = []
    predict_agent_video_label = []
    predict_agents_label = []
    predict_avg_label = []
    path_list = [f'Agent_mcd/storage/{test_type}/false_example/', f'Agent_mcd/storage/{test_type}/process/']
    if save:
        if not os.path.exists(f'Agent_mcd/storage/{test_type}/predict_wrong_{threshold}'):
            os.mkdir(f'Agent_mcd/storage/{test_type}/predict_wrong_{threshold}')
    for p in path_list:
        for i in os.listdir(p):
            with open(p + i, 'r', encoding='utf-8') as file:  
                json_result = json.load(file)  
                if json_result['general_info']['final_rate'] < threshold:
                    predict_label.append(0)
                else:
                    predict_label.append(1)
                scores = []
                labels = []
                tmp = int(re.findall(r'\d+', json_result['controversy_video'][-1]['content'])[-1])
                scores.append(tmp)
                if tmp < threshold:
                    predict_video_label.append(0)
                else:
                    predict_video_label.append(1)
                labels.append(predict_video_label[-1])
                tmp = int(re.findall(r'\d+', json_result['controversy_video_comments'][-1]['content'])[-1])
                scores.append(tmp)
                if tmp < threshold:
                    predict_agent_video_label.append(0)
                else:
                    predict_agent_video_label.append(1)
                labels.append(predict_agent_video_label[-1])
                tmp = int(re.findall(r'\d+', json_result['controversy_comments'][-1]['content'])[-1])
                scores.append(tmp)
                if tmp < threshold:
                    predict_agents_label.append(0)
                else:
                    predict_agents_label.append(1)
                labels.append(predict_agents_label[-1])
                label = stats.mode(labels)[0]
                predict_avg_label.append(label)
                true_label.append(json_result['general_info']['true_label'])
                # if save:
                #     if predict_label[-1] != true_label[-1]:
                #         json_str = json.dumps(json_result, ensure_ascii=False, indent=4)
                #         with open(f'Agent_mcd/storage/{test_type}/predict_wrong_{threshold}/{i}', 'w') as f:
                #             f.write(json_str)
    print(len([i for i in predict_label if i==1]))
    print(len([i for i in predict_label if i==0]))
    print('All: ', metrics(true_label, predict_label))
    print(len([i for i in predict_video_label if i==1]))
    print(len([i for i in predict_video_label if i==0]))
    print('Video self: ', metrics(true_label, predict_video_label))
    print(len([i for i in predict_agent_video_label if i==1]))
    print(len([i for i in predict_agent_video_label if i==0]))
    print('Agent vs video: ', metrics(true_label, predict_agent_video_label))
    print(len([i for i in predict_agents_label if i==1]))
    print(len([i for i in predict_agents_label if i==0]))
    print('Agent vs Agent: ', metrics(true_label, predict_agents_label))
    
if __name__ == "__main__":
    data_types = ['valid', 'test']
    for data_type in data_types:
        for i in range(2,9):
            json_cal_result(test_type=f'simulate_comments_{data_type}_results', threshold=i, save=False)