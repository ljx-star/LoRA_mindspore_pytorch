MASTER_ADDR=localhost \
MASTER_PORT=12355 \
WORLD_SIZE=1 \
RANK=0 \
CUDA_VISIBLE_DEVICES=0 \
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2/e2e/predict.21031_step_final.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt
