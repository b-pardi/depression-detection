from llama_cpp import Llama

from typing import Optional, List, Any, Tuple, Dict
from pathlib import Path
import os
import json

from configs import path_cfg, chat_cfg
from utils import _read_sys_msg

class ChatBot:
    """
    Thin wrapper around llama-cpp chat completions with a running message history.
    """
    def __init__(self, llm, sys_msg=None):
        self.llm = llm
        self.messages = []
        if sys_msg is not None:
            self.messages.append({"role": "system", "content": sys_msg})

    def reset(self, sys_msg=None):
        self.messages = []
        if sys_msg:
            self.messages.append({"role": "system", "content": sys_msg})

    def ask(self, user_text, stream=True):
        """
        append a user message, call the model, append the assistant message, and return it.
        """
        self.messages.append({"role": "user", "content": user_text})

        if stream:
            # inference
            out = []
            response_generator = self.llm.create_chat_completion(
                messages=self.messages,
                #response_format=chat_cfg.chat_fmt,
                temperature=chat_cfg.temperature,
                top_p=chat_cfg.top_p,
                top_k=chat_cfg.top_k,
                max_tokens=chat_cfg.max_output_tokens,
                stream=True
            )

            for event in response_generator:
                delta = event.get('choices', [{}])[0].get('delta', {})
                tkn = delta.get('content', '')
                if tkn:
                    print(tkn, end='', flush=True) # print response live as it generates
                    out.append(tkn)

            print('\n')
            full_response = ''.join(out)
        
        else:
            response = self.llm.create_chat_completion(
                messages=self.messages,
                response_format=chat_cfg.chat_fmt,
                temperature=chat_cfg.temperature,
                top_p=chat_cfg.top_p,
                top_k=chat_cfg.top_k,
                max_tokens=chat_cfg.max_output_tokens,
                stream=False
            )
            full_response = response['choices'][0]['message']['content']
            print(full_response)

        # stash response msg
        self.messages.append({'role': 'assistant', 'content': full_response})
        return full_response
    
    def save_transcript(self, out_fp):
        """
        Save full chat (system + turns) as JSONL (one message per line) for easy reload/grep.
        """
        Path(out_fp.parent).mkdir(parents=True, exist_ok=True)
        with open(out_fp, "w", encoding="utf-8") as chat_log_file:
            for msg in self.messages:
                chat_log_file.write(json.dumps(msg, ensure_ascii=False) + "\n")

    @classmethod
    def from_config(cls, llm: Llama):
        sys_msg = _read_sys_msg(path_cfg.ctx_fp)
        return cls(llm=llm, sys_msg=sys_msg)

            





def query_llm(proj_title, proj_desc, llm):
    # read the system message file -> contains the llm's 'assignment' and examples
    with open(path_cfg.base_data_dir / path_cfg.ctx_fn) as ctx_file:
        sys_msg = ctx_file.read().strip()

    # setup input and context
    prompt = f"Here is the project background, problem(s), and objective(s) for project {proj_title}:\n\n{proj_desc}"
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt}
    ]

    # read in the desired schema (based off the skills_list.json)
    # schema restricts the llm's output to a prestructured json format matching that of 'skills_list.json'
    with open(path_cfg.BASE_DATA_DIR / 'skills_list.json', 'r') as skill_file:
        skill_tree = json.load(skill_file)
    
    skill_tree_schema = {
        "type": "object",
        "properties": {
            "selected_skills": tree_to_schema(skill_tree)
        },
        "required": ["selected_skills"],
        "additionalProperties": False,
    }

    response_format={
        "type": "json_object",
        "schema": skill_tree_schema
    }

    # inference
    response = llm.create_chat_completion(
        messages=messages,
        response_format=response_format,
        #response_format={"type": "json_object"},
        temperature=chat_cfg.temperature,
        top_p=chat_cfg.top_p,
        top_k=chat_cfg.top_k,
        max_tokens=chat_cfg.max_output_tokens,
    )

    try:
        res_text = response['choices'][0]['message']['content']
        res_dict = json.loads(res_text)
        print(f"\n***RESPONSE FOR {proj_title}***\n{res_dict}")
        with open(path_cfg.OUT_DIR / os.path.splitext(path_cfg.MODEL_FN)[0] / f"{proj_title}.json", 'w') as res_file:
            json.dump(res_dict, res_file, indent=2, ensure_ascii=True)

    except Exception as e:
        print(f"Error generating skills for {proj_title}: {e}")
