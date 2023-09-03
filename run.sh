python main.py \
    --strategy mpds \
    --model ../cache/huggingface/models--THUDM--chatglm-6b/snapshots/969290547e761b20fdb96b0602b4fd8d863bbb85/ \
    --data_path ./data/llm_paper_train.json \
    --max_seq_len 128 \
    --batch_size 8 \
    --accumulate_steps 1 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --num_epochs 3  \
    lora