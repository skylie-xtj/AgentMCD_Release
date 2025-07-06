# import re
# from datetime import datetime
# from typing import Any, Dict, List, Optional, Tuple

# from pydantic import BaseModel, Field

# from langchain import LLMChain
# from langchain.base_language import BaseLanguageModel
# from langchain_experimental.generative_agents.memory import GenerativeAgentMemory
# from langchain.prompts import PromptTemplate
# from langchain_experimental.generative_agents import (
#     GenerativeAgent,
#     GenerativeAgentMemory,
# )

class abstract_avatar:
    def __init__(self, args, avatar_id):
        super().__init__()
        self.args = args
        self.avatar_id = avatar_id
        self.use_wandb = args.use_wandb
        self.memory = None
