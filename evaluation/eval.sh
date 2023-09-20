# Evaluation script

# Evaluate 7b model on GSM8k, MATH, and GSM8k_robust
ckpts_dir="./checkpoints/GAIRMath-Abel-7b" # 7b model directory (e.g. "./checkpoints/GAIRMath-Abel-7b")
for DEV_SET in gsm8k math gsm8k_robust
do
sudo CUDA_VISIBLE_DEVICES=0 python -m evaluation.inference --model_dir ${ckpts_dir} --temperature 0.0 --top_p 1.0 --output_file ./outputs/${DEV_SET}/7b.jsonl --dev_set ${DEV_SET} --prompt_type math-single --eval_only True
done

# Evaluate 13b model on GSM8k, MATH, and GSM8k_robust
ckpts_dir="./checkpoints/GAIRMath-Abel-13b" # 13b model directory (e.g. "./checkpoints/GAIRMath-Abel-13b")
for DEV_SET in gsm8k math gsm8k_robust
do
sudo CUDA_VISIBLE_DEVICES=0 python -m evaluation.inference --model_dir ${ckpts_dir} --temperature 0.0 --top_p 1.0 --output_file ./outputs/${DEV_SET}/13b.jsonl --dev_set ${DEV_SET} --prompt_type math-single --eval_only True
done

# Evaluate 70b model on GSM8k, MATH, MathGPT, and GSM8k_robust
ckpts_dir="./checkpoints/GAIRMath-Abel-70b" # 70b model directory (e.g. "./checkpoints/GAIRMath-Abel-70b")
for DEV_SET in gsm8k math mathgpt gsm8k_robust
do
sudo CUDA_VISIBLE_DEVICES=0,1,2,3 python -m evaluation.inference --model_dir ${ckpts_dir} --temperature 0.0 --top_p 1.0 --output_file ./outputs/${DEV_SET}/70b.jsonl --dev_set ${DEV_SET} --prompt_type math-single --eval_only True
done