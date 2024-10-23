import requests
import json

from prompt import text2sql_text
import argparse

# Define the URL of the server endpoint that receives the data
glob_url = "http://localhost:5000/batch_chat"  # Replace 'localhost' if necessary


from typing import (
  Any,
  Dict
)

def get_data_from_server(url, post_body:Dict[str, Any]):
  """
  Args:
    post_body: Dict[str, Any] -> The body of your post request to send to server
      in format like:
      {
        "text": "4月27日到4月29日在重庆住宿两晚; 4月29日再在重庆住宿一晚。"
      }

      And the server reply with following format:
      {
        "response": "xxxx"
      }
  Returns:
   The server's response to your post request
  """


  # The data you want to send
  #data_to_send = {"key": "value"}  # This is just an example, replace with your actual data

  headers = {'Content-type': 'application/json'}  # Defines the content type of the request as JSON

  # Send the POST request and store the response in r
  r = requests.post(url, data=json.dumps(post_body), headers=headers)

  # Print the status code of the server's response
  print("Status Code:", r.status_code)
  print("response: ", r.json())
  return r.json()

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-file", type=str, default="", help="The input file path", required=True)
  parser.add_argument("--output-file", type=str, default="", help="The output file path", required=True)
  parser.add_argument("--batch-size", type=int, default=1, help="Batch size, you can process a group of message once", required=False)
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

  batch_result_list = []
  for start_idx in range(0, len(data_list), args.batch_size):
    batch_text = data_list[start_idx: start_idx + args.batch_size]
    batch_prompt = [text2sql_text.format(input_text=text) for text in batch_text]

    batch_result = get_data_from_server(glob_url, {"text": batch_prompt})

    batch_result_list.append(get_data_from_server(glob_url, {"text": batch_prompt}))

  with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(batch_result_list, f, ensure_ascii=False, indent=2)