#!/bin/sh
export  OUTPUT= "test1"

#python3 modeling.py --data_dir cnn_4ans_100files.txt  --output_dir output/$OUTPUT --max_seq_length --do_train --do_eval --per_gpu_train_batch_size --per_gpu_eval_batch_size --gradient_accumulation_steps --learning_rate --warmup_steps --logging_steps --save_steps --eval_all_checkpoints

python3 modeling.py --data_dir cnn_4ans_100files.txt  --output_dir output/$OUTPUT  --do_train --do_eval
