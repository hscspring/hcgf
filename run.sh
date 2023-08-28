python main.py \
    --strategy mpds \
    --model ../llama.cpp/llama/7B-ChatFlow/ \
    --data_path ./data/llm_paper_train.json \
    --max_seq_len 256 \
    --batch_size 4 \
    --accumulate_steps 32 \
    --num_epochs 10  \
    ia3