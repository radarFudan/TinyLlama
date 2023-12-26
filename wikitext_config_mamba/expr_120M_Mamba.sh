lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=2 \
    --num-nodes=1 \
    ./wikitext_config_mamba/tinyllama_120M_Mamba.py \
        --devices 2  \
        --train_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined \
        --val_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined \
        --resume True
        | tee ./wikitext_config_mamba/expr_120M_Mamba.txt