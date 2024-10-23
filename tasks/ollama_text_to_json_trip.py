import requests
import json

from prompt import text2sql_text
import argparse

# Define the URL of the server endpoint that receives the data
#glob_url = "http://localhost:5000/batch_chat"  # Replace 'localhost' if necessary

"""
curl -X POST http://${host}:11434/api/generate -d '{
  "model": "qwen2:latest",
  "prompt":"write an bash script to find maximum sized file in a directory recursively"
 }'
"""

from typing import (
  Any,
  Dict
)

def get_ollama_url(args: argparse.Namespace):
  return f"http://{args.ollama_host}:11434/api/generate"

"""
import requests

def handle_chunked_response(url, data):
    with requests.post(url, data=data, stream=True) as response:
        if response.ok:
            for chunk in response.iter_content(chunk_size=None):
                # Process each chunk here
                print(chunk.decode())  # Assuming the response is text
        else:
            print(f"Error: {response.status_code}")

# Example usage
url = "https://your-api-endpoint"
data = {"key1": "value1", "key2": "value2"}
handle_chunked_response(url, data)
"""

def get_chunked_response(url, post_body:Dict[str, Any]):
  """
  """
  text_piece_list = []
  with requests.post(url, json=post_body, stream=True) as response:
    if response.ok:
        for chunk_bytes in response.iter_content(chunk_size=None):
          # Process each chunk here
          if isinstance(chunk_bytes, bytes):
            chunk_str = chunk_bytes.decode()
          else:
            chunk_str = chunk_bytes
          # Each recevied chunk like this:
          # {"model":"qwen2:latest","created_at":"2024-10-23T09:46:00.518196Z","response":"\",\n","done":false}
          try:
            chunk_dict = json.loads(chunk_str)
          except json.JSONDecodeError:
            continue
          try:
            text_piece_list.append(chunk_dict["response"])
          except KeyError:
            continue
    else:
      print(f"Error: {response.status_code}")
  return {"response": "".join(text_piece_list)}

def create_payload(args:argparse.Namespace, full_prompt:str):
  """
  Create the json as the payload of a post request
  Args:
    full_prompt: text with model required prefix or suffix
  
  The demo of curl request sent to ollama service:

  ```bash
  curl -X POST http://${host}:11434/api/generate -d '{
    "model": "qwen2:latest",
    "prompt":"write an bash script to find maximum sized file in a directory recursively"
  }'
  ```
  """
  return {
    "model": args.model_name,
    "prompt": full_prompt 
  }

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-file", type=str, default="", help="The input file path", required=True)
  parser.add_argument("--output-file", type=str, default="", help="The output file path", required=True)
  parser.add_argument("--batch-size", type=int, default=1, help="Batch size, you can process a group of message once", required=False)
  parser.add_argument("--ollama-host", type=str, default="localhost", help="Host name or IP, in which ollama service was launched", required=True)
  parser.add_argument("--model-name", type=str, default="", help="model field requred by the request for ollama service", required=True)
  args = parser.parse_args()
  return args

import os
if __name__ == "__main__":

  args = get_args()

  if not os.path.isfile(args.input_file):
    raise FileNotFoundError("The input file does not exist")
  
  data_list = []
  with open(args.input_file, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if not line: 
        continue
      data_list.append(line)
  
  url = get_ollama_url(args)

  assert args.batch_size == 1

  batch_result_list = []
  for start_idx in range(0, len(data_list), args.batch_size):
    batch_text = data_list[start_idx: start_idx + args.batch_size]
    batch_prompt = [text2sql_text.format(input_text=text) for text in batch_text]
    
    if len(batch_prompt) == 1:
      prompt = batch_prompt[0]
    else:
      raise NotImplementedError("No support for batch now")
    
    payload = create_payload(args, prompt) 
    #batch_result_list.append(get_data_from_server(url, payload))
    batch_result_list.append(get_chunked_response(url, payload))

  with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(batch_result_list, f, ensure_ascii=False, indent=2)