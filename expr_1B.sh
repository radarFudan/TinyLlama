CUDA_VISIBLE_DEVICES=1 lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=1 \
    --num-nodes=1 \
    pretrain/tinyllama_WK1B.py --devices 1  --train_data_dir data/wikitext  --val_data_dir data/wikitext --use_wikitext True --resume True
