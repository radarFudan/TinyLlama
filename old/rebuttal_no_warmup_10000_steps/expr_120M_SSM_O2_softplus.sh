lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=1 \
    --num-nodes=1 \
    ./rebuttal_no_warmup_10000_steps/tinyllama_120M_SSM_O2_softplus.py --devices 1 --train_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined