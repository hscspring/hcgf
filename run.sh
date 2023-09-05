python main.py \
    --strategy mpds \
    --model ../cache/huggingface/models--Qwen--Qwen-7B/snapshots/622224017e432fca6410ec2f4d96048b3dd8bc87/ \
    --data_path ./data/llm_paper_train.json \
    --max_seq_len 200 \
    --batch_size 8 \
    --accumulate_steps 1 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --num_epochs 2  \
    lora