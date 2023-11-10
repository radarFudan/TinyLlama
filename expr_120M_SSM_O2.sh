lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=4 \
    --num-nodes=1 \
    pretrain/tinyllama_120M_SSM_O2.py --devices 4 --train_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined