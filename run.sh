python main.py \
    --strategy mpds \
    --model ../cache/huggingface/models--THUDM--chatglm2-6b/snapshots/b1502f4f75c71499a3d566b14463edd62620ce9f/ \
    --data_path ./data/llm_paper_train.json \
    --max_seq_len 256 \
    --batch_size 4 \
    --accumulate_steps 32 \
    --num_epochs 10  \
    ia3