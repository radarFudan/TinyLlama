lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    ./wikitext_config/tinyllama_120M_SSM_O2_exp_safari_adam0999.py --devices 8  --train_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined --val_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined | tee ./wikitext_config/expr_120M_SSM_O2_exp_safari_adam0999.txt