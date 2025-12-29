export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:True
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:True
while true; do
    accelerate launch 3.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium" --teacher_model_name_or_path="/workspace/lora切/wai_v140_A4_21000.safetensors" --train_data_dir="/workspace/lora切/image_balance_2024_webp" --output_dir="sd3" --train_batch_size=1 --num_train_epochs=10 --lr_num_cycles=3 --checkpointing_steps=8000 --validation_steps=4000 --lr_warmup_steps=200 --gradient_checkpointing --mixed_precision="bf16" --learning_rate=4e-6 --lr_scheduler=cosine_with_restarts --drop_tag_rate=0.1 --drop_text_rate=0.1 --drop_char_feature_rate=0.6 --resume_from_checkpoint=latest --sigmas_scale=0.9 --optimizer=muon --inference_steps=4 --use_teacher_text_encoder
    sleep 2
done
