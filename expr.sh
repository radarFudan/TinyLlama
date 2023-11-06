lightning run model \
    --node-rank=0  \
    --main-address=172.16.101.5 \
    --accelerator=cuda \
    --devices=1 \
    --num-nodes=1 \
    pretrain/tinyllama.py --devices 1 --train_data_dir data/wikitext  --val_data_dir data/wikitext --use_wikitext True