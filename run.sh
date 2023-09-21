python main.py \
    --strategy mpds \
    --model ../cache/huggingface/models--Qwen--Qwen-7B-Chat/snapshots/90605029edf5e14988f7d7ef4eb4c76c795c6251/ \
    --data_path ./data/llm_paper_train.json \
    --max_seq_len 200 \
    --batch_size 8 \
    --accumulate_steps 1 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --num_epochs 2  \
    lora \
    --lora_r 8 