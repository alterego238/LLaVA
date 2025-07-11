export CUDA_VISIBLE_DEVICES=1,2
python3 -m accelerate.commands.launch \
    --num_processes=2 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks gqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme \
    --output_path ./logs/