pip install -U "jax==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install jax
lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=2 \
    --num-nodes=1 \
    pretrain/tinyllama_SSM.py --devices 2 --train_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined