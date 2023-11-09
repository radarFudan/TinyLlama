lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=2 \
    --num-nodes=1 \
    pretrain/tinyllama_large_SSM_Order1.py --devices 2 --train_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined