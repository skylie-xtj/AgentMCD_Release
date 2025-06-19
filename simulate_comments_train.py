import os
import shutil
import warnings
warnings.filterwarnings("ignore")
import wandb
from parse import parse_args
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from arena import Arena
from tqdm import tqdm
from utils import *


def prepare_dir():
    """
    make dirs for storage
    """
    def ensureDir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    ensureDir(f'AgentMCD_Release/storage/{args.simulation_name}_{args.test_num}')
    ensureDir(f'AgentMCD_Release/storage/{args.simulation_name}_{args.test_num}' + "/running_logs")
    ensureDir(f'AgentMCD_Release/storage/{args.simulation_name}_{args.test_num}' + "/false_example")
    ensureDir(f'AgentMCD_Release/storage/{args.simulation_name}_{args.test_num}' + "/process")
    if os.path.exists(f'{args.simulation_name}_{args.test_num}' + "/system_log.log"):
        os.remove(f'{args.simulation_name}_{args.test_num}' + "/system_log.log")
    if os.path.exists(f'{args.simulation_name}_{args.test_num}' + "/running_logs/input_prompts.log"):
        os.remove(f'{args.simulation_name}_{args.test_num}' + "/running_logs/input_prompts.log")
    if os.path.exists(f'{args.simulation_name}_{args.test_num}' + "/running_logs/generation_results.log"):
        os.remove(f'{args.simulation_name}_{args.test_num}' + "/running_logs/generation_results.log")
        
def get_data(data_path='AgentMCD_Release/dataset/metadata_test.json'):
    """
    Input:data_path: path to data file
    Output:result:list of dict : {'comments':"*********",'video_des':"*************"}
    用来获取用于测试的数据
    """
    result = []
    path_list = os.listdir(f'{args.simulation_name}_{args.test_num}/false_example/') + os.listdir(f'{args.simulation_name}_{args.test_num}/process/')
    path_list = [i.replace('.json', '') for i in path_list]
    with open(data_path, "r", encoding="utf-8-sig") as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            temp = {}
            data = json.loads(line)
            if data['video_id'] in path_list: continue
            comments = ""
            for i in data["comments"].keys():
                comments += re.sub(
                    r"@\w+\s?", "", data["comments"][i]["comments"]
                )  # comment_content
            comments = comments.replace("\r", " ")
            temp["comments"] = comments
            temp['video_des'] = data['video_des']
            temp['video_id'] = data['video_id']
            temp['label'] = data['controversy']
            temp['keywords'] = data['keywords']
            temp['title'] = data['title']
            result.append(temp)
    return result

def run(args, data):
    arena_ = Arena(args)
    arena_.execute(data)

if __name__ == '__main__':
    args = parse_args()
    fix_seeds(args.seed) # set random seed    
    if(args.use_wandb):
        wandb.init(
            project = "sandbox",
            name = args.simulation_name,
            group = args.dataset
        )
    data_types = ['valid', 'test']
    for data_type in data_types:
        prepare_dir()
        data_path = f'AgentMCD_Release/dataset/metadata_{data_type}.json'
        datas = get_data(data_path)
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for i, data in enumerate(datas):
                futures.append(executor.submit(run, args, data))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception occurred: {e}")
        print('Simulation finished!')
        if(args.use_wandb):
            wandb.finish()
