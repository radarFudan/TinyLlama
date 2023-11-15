# CUDA_VISIBLE_DEVICES=0 lightning run model \
#     --node-rank=0  \
#     --accelerator=cuda \
#     --devices=1 \
#     --num-nodes=1 \
#     pretrain/tinyllama_WK120M.py --devices 1  --train_data_dir data/wikitext  --val_data_dir data/wikitext --use_wikitext True | tee ./pretrain/tinyllama_WK120M.txt

CUDA_VISIBLE_DEVICES=0 lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=1 \
    --num-nodes=1 \
    pretrain/tinyllama_WK120M_2epochs.py --devices 1  --train_data_dir data/wikitext  --val_data_dir data/wikitext --use_wikitext True | tee ./pretrain/tinyllama_WK120M_2epochs.txt

