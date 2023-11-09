import random
from tqdm import tqdm
from random import shuffle
import os


def split_file_to_chunks(file_path, target_folder):
    chunk_idx = 0
    lines = open(file_path, "r", encoding="utf8").readlines()
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        os.makedirs(f"{target_folder}/train")
        os.makedirs(f"{target_folder}/valid")
    shuffle(lines)
    # split into train / valid
    border = int(len(lines) * 0.95)
    for i in tqdm(range(0, len(lines), 100000)):
        chunk = lines[i:i+100000]
        prefix = "train" if i < border else "valid"
        with open(f"{target_folder}/{prefix}/chunk_{chunk_idx}.jsonl", "w", encoding="utf8") as f:
            f.writelines(chunk)
        chunk_idx += 1
        
if __name__ == "__main__":
    # split_file_to_chunks("/data/hf_dataset/en_merge_sample.jsonl",
    #                      "/data/hf_dataset/en_merge_sample")
    # split_file_to_chunks("../../slimpajama_en_60b.jsonl",
    #                     "/home/aiops/liuqian/TinyLlama/hf_dataset/slimpajama_en_60b")
    split_file_to_chunks("/home/aiops/wangsd/mix_sample.jsonl",
                        "/home/aiops/wangsd/TinyLlama/data/mix_sample")