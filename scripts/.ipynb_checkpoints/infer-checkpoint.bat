@echo off
set count=%1
set api_token=%2
set level=%3

for /l %%i in (0,1,%count%) do (
    start python scripts/classify_gpt.py --chunk_id %%i --chunk_size %count% --API_TOKEN %api_token% --pred_cause_level %level%
)

ping 127.0.0.1 -n 5 > nul

python scripts/classify_gpt.py --chunk_id 0 --chunk_size %count% --pred_cause_level %level% --concat True
