export model_dir="./pretrained_checkpoints"
export task="cola"
export data_dir="./glue/$task"
export output_dir="./result/$task"
MASTER_ADDR=localhost \
MASTER_PORT=12355 \
WORLD_SIZE=1 \
RANK=0 \
CUDA_VISIBLE_DEVICES=0 \
python examples/text-classification/run_glue.py \
--model_name_or_path $model_dir/DeBERTa-v3-sm \
--do_train \
--do_eval \
--train_file $data_dir/train.csv \
--test_file $data_dir/test.csv \
--validation_file $data_dir/dev.csv \
--max_seq_length 64 \
--per_device_train_batch_size 2 \
--learning_rate 1.3e-4 \
--num_train_epochs 10 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 20 \
--save_strategy steps \
--save_steps 20 \
--warmup_steps 100 \
--cls_dropout 0.1 \
--apply_lora \
--lora_r 16 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0 \
--use_deterministic_algorithms
#--task_name cola \