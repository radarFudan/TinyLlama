lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=1 \
    --num-nodes=1 \
    ./wikitext_config_mamba/tinyllama_120M_PureMamba.py \
        --devices 1  \
        --train_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined \
        --val_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined \
        | tee ./wikitext_config_mamba/expr_120M_PureMamba.txt