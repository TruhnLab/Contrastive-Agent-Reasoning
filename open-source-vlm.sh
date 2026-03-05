#!/bin/bash
#SBATCH --job-name=cxr
#SBATCH --time=5-40:00:00
#SBATCH --nodelist=hawking
#SBATCH --partition=radcluster
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G

nvidia-smi
source ~/vlm/bin/activate


# For dermoscopy data
python derm_agent_script.py --model_name medvlm --output_path derm_medvlm.csv
python derm_agent_script_judge.py --model_name glm --output_path derm_glm_care.csv
python derm_agent_script_care.py --model_name qwen-32 --output_path derm_qwen_32_care.csv
python derm_agent_script.py --model_name gemma --output_path derm_gemma.csv
python derm_agent_script.py --model_name qwen --output_path derm_qwen.csv


# For chest x-ray data
python cxr_agent_script.py --model_name qwen --output_path cxr_qwen.csv
python cxr_agent_script_care.py --model_name qwen --output_path cxr_qwen_care.csv
# ...