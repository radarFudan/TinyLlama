version=1

lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    pretrain/tinyllama_v${version}.py --devices 8 --train_data_dir /home/aiops/wangsd/TinyLlama/data/the_pile_deduplicated_EleutherAI_combined | tee scripts_logs/expr_v${version}.txt

# Match the hyperparameters as the mamba. 
