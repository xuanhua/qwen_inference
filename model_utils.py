
# Use a single file to handle various model loading and initialization of tokenizers
# TODO
import torch
from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM,
  GenerationConfig
)

from config import (
  QWEN_1_8B_PATH,
  QWEN_72B_INT4_PATH,
  QWEN_72B_INT8_PATH
)

def _only_has_one_8gb_gpu():
    """
    Check if there is only one GPU and its vram is 8GB of memory.
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus != 1:
        return False
    total_memory = torch.cuda.get_device_properties(0).total_memory
    gpu_mem_gb = total_memory / 1024**3
    if gpu_mem_gb > 7 and gpu_mem_gb <= 8:
        return True
    else:
        return False

def get_pretrained_model(model_path:str, tokenizer: AutoTokenizer, *args, **kwargs):
  """
  Call get_pretrained_tokenizer() first, then call this function.
  """
  assert model_path
  assert args == ()

  # TODO: once needs customization of some keyword argument, we can use these part of code
  #pad_token_id = kwargs.pop("pad_token_id", tokenizer.pad_token_id)
  #device_map = kwargs.pop("device_map", "auto")
  #trust_remote_code = kwargs.pop("trust_remote_code", True)

  if model_path == QWEN_1_8B_PATH and _only_has_one_8gb_gpu():
    model = AutoModelForCausalLM.from_pretrained(
      model_path,
      pad_token_id=tokenizer.pad_token_id,
      device_map="auto",
      trust_remote_code=True,
      torch_dtype=torch.float16, # Use FP16 for faster inference on GPUs with bf16 support.
    ).eval()
  else:
    model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True
            ).eval()
  model.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=tokenizer.pad_token_id)
  return model


def get_pretrained_tokenizer(model_path, *args, **kwargs):
  """
  Pass extra arguments to xxxx.from_pretrained() by keyword arguments
  """
  assert model_path
  assert args == ()

  trust_remote_code = kwargs.pop("truct_remote_code", True)
  pad_token = kwargs.pop("pad_token", '<|extra_0|>')
  eos_token = kwargs.pop("eos_token", '<|endoftext|>')
  padding_side = kwargs.pop("padding_side", 'left')

  return AutoTokenizer.from_pretrained(model_path,
                                       trust_remote_code=trust_remote_code,
                                       pad_token=pad_token,
                                       eos_token=eos_token,
                                       padding_side=padding_side)

if __name__ == "__main__":
  pass