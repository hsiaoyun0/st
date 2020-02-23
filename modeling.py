import argparse
import logging
import os
import random
import numpy as np

import torch
from transformers import *

from utils import ReadFile

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()

    #Required parameters
    parser.add_argument("--data_dir", type=str, help="The input dir.")
    parser.add_argument("--output_dir", type=str, help="The output dir")

    #Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--per_gpu_train_batch_size",default=8, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--logging_steps", default=500, type=int)
    parser.add_argument("--save_steps", default=500, type=int)
    parser.add_argument("--eval_all_checkpoints",action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    #Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    #Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    
    #set_seed
    set_seed(args)

    #Prepare cnn/dailymail task
    dataset = ReadFile(args.data_dir)
    
    #Load 

if __name__ == "__main__":
    main()
