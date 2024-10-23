#!/bin/bash

echo "Make sure you run this script from the project home directory"

export PYTHONPATH=./:$PYTHONPATH

python tasks/text_to_json_trip.py   --input-file data/text_to_jsontrip.txt   --output-file  data/result-of-jsontrip.json --batch-size 2
