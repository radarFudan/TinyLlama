version=2

lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    pretrain/tinyllama_v${version}_sanitycheck.py \
    --devices 8 --train_data_dir /home/aiops/wangsd/TinyLlama/data/the_pile_deduplicated_EleutherAI_combined | tee scripts_logs/expr_v${version}.txt

# Make sure it's still similar to version 2. 
