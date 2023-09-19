python main.py \
    --strategy mpds \
    --model ../llama.cpp/llama/7B-ChatFlow/ \
    --data_path ./data/llm_paper_train.json \
    --max_seq_len 200 \
    --batch_size 8 \
    --accumulate_steps 1 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --num_epochs 2  \
    lora \
    --lora_r 8 