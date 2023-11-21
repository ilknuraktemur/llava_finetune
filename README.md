# llava_finetune
bash run.sh
bash /workspace/ilknur/LLaVA/scripts/v1_5/finetune_task_lora.sh #fine-tune script
python /workspace/ilknur/LLaVA/scripts/merge_lora_weights.py --model-path "/workspace/ilknur/LLaVA/checkpoints/llava-v1.5-13b-task-lora" --model-name "ll
ava-v1.5-13b" --save-model-path "/workspace/ilknur" 
python inference.py