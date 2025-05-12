import json
import os
import pickle
import random
import re
import time
from datetime import datetime
import torch
from prompt.prompt import *
from simulation.avatar import Avatar
from termcolor import cprint
from utils import metrics
from llm.glm4_9b import *


class Arena:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.val_users = args.val_users
        self.val_ratio = args.val_ratio
        self.simulation_name = args.simulation_name
        self.test_num = args.test_num
        self.device = torch.device(args.cuda)
        self.max_workers = args.max_workers
        self.execution_mode = args.execution_mode
        self.llm_type = args.llm_type
        self.simulation_name = args.simulation_name
        self.persona_df = pickle.load(
            open("AgentMCD_Release/dataset/agent_role.pkl", "rb")
        )
        self.storage_base_path = f"AgentMCD_Release/storage/{args.simulation_name}_{args.test_num}"
        self.save_file = {}
        self.id_asr = pickle.load(
            open("AgentMCD_Release/dataset/whisper_id_asr.pkl", "rb")
        )
        self.id_ocr = pickle.load(
            open("AgentMCD_Release/dataset/clean_ocr_youdao.pkl", "rb")
        )
        self.orig_des = pickle.load(
            open("AgentMCD_Release/dataset/id_des_new.pkl", "rb")
        )
        self.t_label = []
        self.t_predict = []
        self.start_time = time.time()
    
    def save_file_to_json(self, id):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        if not os.path.exists(self.storage_base_path + '/process'):
            os.mkdir(self.storage_base_path + '/process')
        if not os.path.exists(self.storage_base_path + '/false_example'):
            os.mkdir(self.storage_base_path + '/false_example')
        if self.save_file["general_info"]['true_label'] != self.save_file["general_info"]['predicself.t_label']:
            save_file_path = os.path.join(self.storage_base_path + '/false_example', f"{id}.json")
        else:
            save_file_path = os.path.join(self.storage_base_path + '/process', f"{id}.json")
        self.save_file["general_info"]['end_time'] = current_time
        json_str = json.dumps(self.save_file, ensure_ascii=False, indent=4)
        with open(save_file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
  
        
    def initialize_all_avatars(self, id):
        """
        initialize all avatars
        """
        self.avatars = {}
        i = 0
        agent_role = []
        for k in self.persona_df[id].keys():
            self.avatars[i] = Avatar(self.args, i, k + '<' + self.persona_df[id][k] + '>')
            agent_role.append(k + '<' + self.persona_df[id][k]+ '>')
            i += 1
            # if i == 4: break
        return agent_role

    def write_log(self, log_type, log, color=None, attrs=None, print=False):
        if log_type == 'input':
            log_file = f"AgentMCD_Release/storage/{self.simulation_name}/running_logs/input_prompts.log"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(str(log) + '\n')
                f.flush()
            if(print):
                cprint(log, color=color, attrs=attrs)
        if log_type == 'output':
            log_file = f"AgentMCD_Release/storage/{self.simulation_name}/running_logs/generation_results.log"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(str(log) + '\n')
                f.flush()
            if(print):
                cprint(log, color=color, attrs=attrs)
    
    def execute(self, data):
        """
        bigin simulation
        """
        self.save_file = {}
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.save_file["general_info"] = {
            "start_time": current_time, 
            "llm_type": self.llm_type,  
            "video_id": data['video_id'], 
            "true_label": int(data['label'])
        }
        # refine description
        sys_prompt, user_prompt = prepare_refine_des_prompt(data['keywords'], data['title'], '', '', '')
        self.save_file["video_des"] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": data['video_des']}
        ]
        # agent description
        agent_num = 5
        sys_prompt, user_prompt = prepare_agent_role_prompt(data['video_des'], agent_num)
        agent_role = self.initialize_all_avatars(data['video_id'])
        for i in range(len(agent_role)):
        # for i in range(agent_num):
            self.avatars[i] = Avatar(self.args, i, agent_role[i])
        self.save_file["agent_role"] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": agent_role}
        ]
        # generate responses
        responses = []
        self.save_file["avatar_response"] = []   
        for avatar_id in range(len(self.avatars)):
            avatar_ = self.avatars[avatar_id]
            sys_prompt, user_prompt = prepare_comment_prompt(avatar_.role, data['video_des'])
            results = generate_response_vllm(self.llm_type, sys_prompt=sys_prompt, user_prompt=user_prompt)
            responses.append(results[-1]['content'])
            self.save_file["avatar_response"].append(results)   
        # more respons es
        self.save_file["avatar_more_response"] = []   
        for avatar_id in range(len(self.avatars)):
            avatar_ = self.avatars[avatar_id]
            sys_prompt, user_prompt = prepare_more_comment_prompt(avatar_.role, responses)
            results = generate_response_vllm(self.llm_type, sys_prompt=sys_prompt, user_prompt=user_prompt)
            self.save_file["avatar_more_response"].append(results)   
        summary = []
        # 0: controversy of video
        sys_prompt, user_prompt = prepare_video_controversy_prompt(data['video_des'])
        results = generate_response_vllm(self.llm_type, sys_prompt=sys_prompt, user_prompt=user_prompt)
        summary.append(results[-1]['content'])
        self.save_file["controversy_video"] = results        
        # 1: controversy of video and comments
        sys_prompt, user_prompt = prepare_video_comments_controversy_prompt(data['video_des'], responses)
        results = generate_response_vllm(self.llm_type, sys_prompt=sys_prompt, user_prompt=user_prompt)
        summary.append(results[-1]['content'])
        self.save_file["controversy_video_comments"] = results        
        # 2: controversy of comments
        sys_prompt, user_prompt = prepare_comments_controversy_prompt(responses)
        results = generate_response_vllm(self.llm_type, sys_prompt=sys_prompt, user_prompt=user_prompt)
        summary.append(results[-1]['content'])
        self.save_file["controversy_comments"] = results        
        # comprehensive
        sys_prompt, user_prompt = prepare_final_prompt(summary)
        results = generate_response_vllm(self.llm_type, sys_prompt=sys_prompt, user_prompt=user_prompt)
        self.save_file["controversy_final"] = results
        # 使用正则表达式查找所有数字
        result = int(re.findall(r'\d+', results[-1]['content'].split('\n争议评分')[-1])[0])
        self.save_file["general_info"]['final_rate'] = result
        if result < 5:
            self.t_predict.append(0)
            self.save_file["general_info"]['predicself.t_label'] = 0
        else:
            self.t_predict.append(1)
            self.save_file["general_info"]['predicself.t_label'] = 1
        self.t_label.append(data['label'])
        results = metrics(self.t_label, self.t_predict)
        self.save_file_to_json(data['video_id'])
