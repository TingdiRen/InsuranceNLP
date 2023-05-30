#!/bin/bash
count=$1
api_token=$2
level=$3
for ((i=0; i<$count; i++)); do
    python classify_gpt.py --chunk_id $i --chunk_size $count --API_TOKEN $api_token --pred_cause_level $level &
done
wait
python classify_gpt.py --chunk_id 0 --chunk_size $count --pred_cause_level $level --concat True