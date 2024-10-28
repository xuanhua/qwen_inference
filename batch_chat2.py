
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
#from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids

from Qwen_1_8B_Chat.qwen_generation_utils import (
  HistoryType,
  make_context,
  decode_tokens,
  get_stop_words_ids,
  StopWordsLogitsProcessor
)

from config import (
  QWEN_1_8B_PATH,
  QWEN_72B_INT4_PATH
)

#pretrained_model_path = QWEN_1_8B_PATH

# To generate attention masks automatically, it is necessary to assign distinct
# token_ids to pad_token and eos_token, and set pad_token_id in the generation_config.
#tokenizer = AutoTokenizer.from_pretrained(
#    pretrained_model_path,
#    pad_token='<|extra_0|>',
#    eos_token='<|endoftext|>',
#    padding_side='left',
#    trust_remote_code=True
#)

#if pretrained_model_path == QWEN_1_8B_PATH:
#  model = AutoModelForCausalLM.from_pretrained(
#      pretrained_model_path,
#      pad_token_id=tokenizer.pad_token_id,
#      device_map="auto",
#      trust_remote_code=True,
#      torch_dtype=torch.float16, # Use FP16 for faster inference on GPUs with bf16 support.
#  ).eval()
#else:
#  model = AutoModelForCausalLM.from_pretrained(
#                pretrained_model_path,
#                device_map="auto",
#                trust_remote_code=True
#            ).eval()
#model.generation_config = GenerationConfig.from_pretrained(pretrained_model_path, pad_token_id=tokenizer.pad_token_id)

long_input = """
你是一个行程安排助理，你负责将用户的出行计划的语言描述请求转化为结构化的Json信息，

用户的出行计划会涉及到：坐飞机、坐火车，住酒店以及坐出租车等申请项；

各种申请项中你需要分别识别如下的字段：

飞机：
出发城市，如果用户没有提及，默认城市为北京
目的城市
出发时间

火车
出发城市，如果用户没有提及，默认城市为北京
目的城市
出发时间

酒店：
入住城市
入住日期，为到达该城市的日期
离店日期，为离开该城市的日期
如果需要酒店，入住城市需要包含所有需要停留超过一天的城市；如果有多个入住城市，将其拆开展示

用车：
用车城市
开始日期
结束日期
用车次数

例1：
输入：4月19日申请从广州到重庆的国内机票,同时申请在重庆当天用车2次。
输出：
```json
[
  {
    "目的城市": "重庆市",
    "出发城市": "广州市",
    "出发日期": "2022-04-19",
    "请求类型": "国内机票"
  },
  {
    "结束日期": "2022-04-19",
    "开始日期": "2022-04-19",
    "用车城市": [
      "重庆市"
    ],
    "用车次数": 2,
    "请求类型": "用车"
  }
]
```

例2
输入：4月23日申请从丽江到成都的国内机票，然后从成都到乌鲁木齐的国内机票。
输出：
```json
[
  {
    "目的城市": "成都市",
    "出发城市": "丽江市",
    "出发日期": "2022-04-23",
    "请求类型": "国内机票"
  },
  {
    "目的城市": "乌鲁木齐市",
    "出发城市": "成都市",
    "出发日期": "2022-04-23",
    "请求类型": "国内机票"
  }
]
```

例3
输入：4月21日乘坐火车从吉安到湖州；4月21日至23日入住湖州市酒店两晚。
输出：
```json
[
  {
    "出发城市": "吉安市",
    "目的城市": "湖州市",
    "出发日期": "2022-04-21",
    "请求类型": "火车"
  },
  {
    "入住日期": "2022-04-21",
    "目的城市": "湖州市",
    "离店日期": "2022-04-23",
    "请求类型": "酒店"
  }
]
```

例4
输入：3月15日坐火车从珠海到广州,然后再从广州到深圳,最后从深圳到赣州,当天在赣州住两晚; 期间一共申请2次用车; 3月17日从赣
州坐火车到长沙。
输出： 
```json
[
  {
    "出发城市": "珠海市",
    "目的城市": "广州市",
    "出发日期": "2022-03-15",
    "请求类型": "火车"
  },
  {
    "出发城市": "广州市",
    "目的城市": "深圳市",
    "出发日期": "2022-03-15",
    "请求类型": "火车"
  },
  {
    "出发城市": "深圳市",
    "目的城市": "赣州市",
    "出发日期": "2022-03-15",
    "请求类型": "火车"
  },
  {
    "入住日期": "2022-03-15",
    "目的城市": "赣州市",
    "离店日期": "2022-03-17",
    "请求类型": "酒店"
  },
  {
    "结束日期": "2022-03-17",
    "开始日期": "2022-03-15",
    "用车城市": [
      "珠海市",
      "赣州市",
      "长沙市"
    ],
    "用车次数": 2,
    "请求类型": "用车"
  },
  {
    "出发城市": "赣州市",
    "目的城市": "长沙市",
    "出发日期": "2022-03-17",
    "请求类型": "火车"
  }
]
```

输入：4月28日申请北京飞广州的飞机，需要提前一天从石家庄坐高铁赶往北京。
输出：
"""
from typing import (
  Dict,
  List,
  Union
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

def batch_chat_impl(model: AutoModelForCausalLM,
               tokenizer: AutoTokenizer, 
               batch_input: Union[str, List[str]]):
  """
  """
  assert batch_input
  if isinstance(batch_input, str):
    batch_input = [batch_input]

  #all_raw_text = ["我想听你说爱我。", "今天我想吃点啥，甜甜的，推荐下", "我马上迟到了，怎么做才能不迟到", long_input]
  batch_raw_text = []
  for input_text in batch_input:
      raw_text, _ = make_context(
          tokenizer,
          input_text,
          system="You are a helpful assistant.",
          max_window_size=model.generation_config.max_window_size,
          chat_format=model.generation_config.chat_format,
      )
      batch_raw_text.append(raw_text)

  batch_input_ids = tokenizer(batch_raw_text, padding='longest')
  batch_input_ids = torch.LongTensor(batch_input_ids['input_ids']).to(model.device)
  batch_out_ids = model.generate(
      batch_input_ids,
      return_dict_in_generate=False,
      generation_config=model.generation_config
  )
  padding_lens = [batch_input_ids[i].eq(tokenizer.pad_token_id).sum().item() for i in range(batch_input_ids.size(0))]

  batch_response = [
      decode_tokens(
          batch_out_ids[i][padding_lens[i]:],
          tokenizer,
          raw_text_len=len(batch_raw_text[i]),
          context_length=(batch_input_ids[i].size(0)-padding_lens[i]),
          chat_format="chatml",
          verbose=False,
          errors='replace'
      ) for i in range(len(batch_raw_text))
  ]
  print(batch_response)

  return batch_response

  #response, _ = model.chat(tokenizer, "我想听你说爱我。", history=None)
  #print(response)
  #
  #response, _ = model.chat(tokenizer, "今天我想吃点啥，甜甜的，推荐下", history=None)
  #print(response)
  #
  #response, _ = model.chat(tokenizer, "我马上迟到了，怎么做才能不迟到", history=None)
  #print(response)