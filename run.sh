python main.py \
    --strategy fsdp_zero3 \
    --model ../cache/huggingface/models--THUDM--chatglm-6b/snapshots/619e736c6d4cd139840579c5482063b75bed5666/ \
    --data_path ./data/llm_paper_train.json \
    --max_seq_len 200 \
    --batch_size 2 \
    --accumulate_steps 1 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --num_epochs 2  \
    lora