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
--model_name_or_path $model_dir/RoBERTa-base \
--train_file $data_dir/train.csv \
--test_file $data_dir/test.csv \
--validation_file $data_dir/dev.csv \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 32 \
--learning_rate 4e-4 \
--num_train_epochs 80 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--apply_lora \
--lora_r 8 \
--lora_alpha 16 \
--seed 0 \
--weight_decay 0.1
#--task_name cola \
