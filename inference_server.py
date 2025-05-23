from typing import (
    List,
    Optional,
    Tuple,
    Union
)

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map

from config import (
    QWEN_1_8B_PATH,
    QWEN_72B_INT4_PATH,
    QWEN_72B_INT8_PATH
)

from model_utils import (
    get_pretrained_tokenizer,
    get_pretrained_model
)

#from batch_chat import batch_chat as custom_batch_chat
from batch_chat2 import batch_chat_impl

import argparse
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", 
                        type = str, 
                        choices=['72b_int4', '72b_int8', '1.8b'], 
                        default = "72b_int4", 
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

        self._tokenizer = get_pretrained_tokenizer(model_path) 
        self._model = get_pretrained_model(model_path, self._tokenizer)
    
    def chat(self, text: str):
        """
        """
        print(f">>>>>>>>>>>>{__file__}: axu_ts={time.time()}")
        response, history = self._model.chat(self._tokenizer, text, history=None)
        return response, history
    
    def batch_chat(self, text:Union[str, List[str]]):
        print(f">>>>>>>>>>>>{__file__}: axu_ts={time.time()}")
        response = batch_chat_impl(self._model, 
                                            self._tokenizer, 
                                            text)
        return response, None

    def batched_generate(self, texts: Union[str, List[str]], batch_size: int = 32):
        """
        Fix this later
        """
        #inputs = self._tokenizer.encode(text, return_tensors='pt')
        #outputs = self._model.generate(inputs, max_length=50)
        #response = self._tokenizer.decode(outputs[0])
        #return response
        pass

def run(model: Qwen72bModel):
    """
    Keep this simple demo in case we need to run the model directly.
    """
    from prompt import text2sql_text 
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
    prompt = data["text"]
    #prompt = text2sql_text.format(input_text=text)

    start_time = time.time()
    response, _  = glob_model.chat(prompt)
    total_secs= time.time() - start_time
    print("Response time: ", total_secs)
    
    return jsonify({"response": response})

@app.route("/batch_chat", methods=["POST"])
def batch_chat():
    data = request.get_json()
    batch_prompt = data["text"]
    start_time = time.time()
    response, _ = glob_model.batch_chat(batch_prompt)
    print(f"Response time: {time.time() - start_time}")

    return jsonify({"response": response})

if __name__ == "__main__":
    """
    start a flask server and wait for user input, then generate response.
    """
    # If you only want to use 1 gpu (24GB vram available) for inference( 1.8b, 7b model could work under such circumstances)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"   
    args = set_args()
    glob_model = Qwen72bModel(args)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
