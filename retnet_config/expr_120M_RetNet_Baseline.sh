lightning run model \
    --node-rank=0  \
    --accelerator=cuda \
    --devices=4 \
    --num-nodes=1 \
     ./retnet_config/tinyllama_120M_RetNet_Baseline.py --devices 4  --train_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined --val_data_dir /home/aiops/wangsd/TinyLlama/data/mix_sample_combined | tee ./retnet_config/expr_120M_RetNet_Baseline.txt

# No polynomial decay / linear decay. Implement a bit later. 
# Currently follow the appendix setting 