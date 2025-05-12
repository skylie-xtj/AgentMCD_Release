# from llm.glm4 import generate_response
# from llm.qwen import generate_response
from llm.glm4_9b import *

# from llm.glm4_flash import generate_response
from prompt.prompt import *
from termcolor import cprint
from simulation.base.abstract_avatar import abstract_avatar

# from llm.gpt import generate_response

class Avatar(abstract_avatar):
    def __init__(self, args, avatar_id, init_property):
        super().__init__(args, avatar_id)
        self.role = init_property
        # self.init_memory()
        self.llm_type = args.llm_type
        self.lang = args.lang
        self.simulation_name = args.simulation_name
         
    def reaction_to_video(self, video_str, api_key):
        """
        Summarize the feelings of the avatar for the video.
        """ 
        sys_prompt, user_prompt = prepare_comment_prompt(self.role, video_str)
        reaction = generate_response_vllm(self.llm_type, api_key=api_key, sys_prompt=sys_prompt, user_prompt=user_prompt)
        return reaction
    
    def reaction_to_other(self, comments, api_key):
        """
        Summarize the feelings of the avatar for the video.
        """ 
        sys_prompt, user_prompt = prepare_more_comment_prompt(self.role, comments)        
        reaction = generate_response_vllm(self.llm_type, api_key=api_key, sys_prompt=sys_prompt, user_prompt=user_prompt)
        return reaction

    def write_log(self, log_type, log, color=None, attrs=None, print=False):
        log_file=f'../storage/{self.simulation_name}/running_logs/input_prompts.log'
        if log_type == 'input':
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(str(log) + '\n')
                f.flush()
            if(print):
                cprint(log, color=color, attrs=attrs)
