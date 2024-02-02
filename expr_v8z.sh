version=8

lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=1 \
    --num-nodes=1 \
    pretrain/tinyllama_v${version}.py \
    --devices 1 --train_data_dir /home/aiops/wangsd/TinyLlama/data/the_pile_deduplicated_EleutherAI_split_combined | tee scripts_logs/expr_v${version}.txt

# Slow version, causal conv should be right, mamba might be wrong, data might be wrong. 
# zero mode