from typing import (
    List,
    Optional,
    Tuple,
    Union
)

from transformers import AutoTokenizer, AutoModelForCausalLM
import time

QWEN_72B_INT4_PATH = "/data/hg_models/Qwen-72B-Chat-Int4"
QWEN_72B_INT8_PATH = "/data/hg_models/Qwen-72B-Chat-Int8"
QWEN_1_8B_PATH = "/data/hg_models/Qwen-1_8B-Chat"

def demo_code():
    model_path = "/data/hg_models/Qwen-72B-Chat-Int8"

    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    ).eval()


    start = time.time()
    response, history = model.chat(tokenizer, "你好", history=None)
    print(response)
    # 你好！很高兴为你提供帮助。

    # Qwen-72B-Chat现在可以通过调整系统指令（System Prompt），实现角色扮演，语言风格迁移，任务设定，行为设定等能力。
    # Qwen-72B-Chat can realize roly playing, language style transfer, task setting, and behavior setting by system prompt.
    response, _ = model.chat(tokenizer, "你好呀", history=None, system="请用二次元可爱语气和我说话")
    print(response)
    # 哎呀，你好哇！是怎么找到人家的呢？是不是被人家的魅力吸引过来的呀~(≧▽≦)/~

    response, _ = model.chat(tokenizer, "My colleague works diligently", history=None, system="You will write beautiful compliments according to needs")
    print(response)
    # Your colleague is a shining example of dedication and hard work. Their commitment to their job is truly commendable, and it shows in the quality of their work. 
    # They are an asset to the team, and their efforts do not go unnoticed. Keep up the great work!

    end = time.time()

    print(f"Total time cost is {end-start} seconds")

import argparse
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", 
                        type = str, 
                        choices=['72b_int4', '72b_int8', '1.8b'], 
                        default = "1.8b", 
                        help = "Model Type: 72b_int4/72b_int8/1.8b", 
                        required=True)
    args = parser.parse_args()
    return args

class Qwen72bModel:
    def __init__(self, args: argparse.Namespace):
        # Note: The default behavior now has injection attack prevention off.
        if args.model_type == "72b_int4":
            model_path = QWEN_72B_INT4_PATH
        elif args.model_type == "72b_int8":
            model_path = QWEN_72B_INT8_PATH
        elif args.model_type == "1.8b":
            model_path = QWEN_1_8B_PATH
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        ).eval()
    
    def chat(self, text: str):
        response, history = self._model.chat(self._tokenizer, text, history=None)
        return response, history
    
    def batched_generate(self, texts: Union[str, List[str]], batch_size: int = 32):
        """
        Fix this later
        """
        #inputs = self._tokenizer.encode(text, return_tensors='pt')
        #outputs = self._model.generate(inputs, max_length=50)
        #response = self._tokenizer.decode(outputs[0])
        #return response
        pass

from prompt import text2sql_text 
def run(model: Qwen72bModel):
    #input_text = "4月27日乘坐火车从郑州到西安；4月27日至29日在西安市住宿两晚。"
    input_text = "4月28日申请北京飞广州的飞机，需要提前一天从石家庄坐高铁赶往北京。"
    prompt = text2sql_text.format(input_text=input_text) 
    response, _ = model.chat(prompt)
    print("Response: ", response)

from flask import Flask, request, jsonify
app = Flask(__name__)


glob_model = None

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    text = data["text"]
    prompt = text2sql_text.format(input_text=text)

    start_time = time.time()
    response, _  = glob_model.chat(prompt)
    total_secs= time.time() - start_time
    print("Response time: ", total_secs)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    """
    start a flask server and wait for user input, then generate response.
    """
    args = set_args()
    glob_model = Qwen72bModel(args)
    app.run(port=5000, debug=True, use_reloader=False)
