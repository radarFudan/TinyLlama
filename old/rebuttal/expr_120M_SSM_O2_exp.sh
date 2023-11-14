lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=2 \
    --num-nodes=1 \
    ./rebuttal/tinyllama_120M_SSM_O2_exp.py --devices 2 --train_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined