
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

from Qwen_1_8B_Chat.qwen_generation_utils import (
  HistoryType,
  make_context,
  decode_tokens,
  get_stop_words_ids,
  StopWordsLogitsProcessor
)

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
  This implementation comes from the official qwen github's README.md, check if if there is any issue here.
  """
  assert batch_input
  if isinstance(batch_input, str):
    batch_input = [batch_input]

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