
from PROMPT import EDEMA_AGENT_PROMPT, PNEUMONIA_AGENT_PROMPT, JUDGE_TEXT_AGENT_PROMPT, JUDGE_IMAGE_TEXT_AGENT_PROMPT, INSTRUCTION_PROMPT
import pandas as pd
import ast
import argparse
import os
import pdb
from tqdm import tqdm



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default='mimic-cxr-edema-pneumonia-dcm.csv',
                        help="Path to the input CSV file")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["phi", "intern", "qwen", "deepseek","gemma","qwen-32","medgemma-27","medvlm","medgemma_15","glm"],
                        help="Which VLM to use")
    parser.add_argument("--output_path", type=str, default='./medgemma_1.5_4b.csv')

    args = parser.parse_args()
    data_prefix='/home/homesOnMaster/zzhao/data/mimic_cxr_care/'
    weights_prefix='/home/homesOnMaster/zzhao/weights/'
    csv_path = args.csv_path
    output_path = args.output_path
    model_name = args.model_name

    Instruction_prompt=INSTRUCTION_PROMPT
    print("Using the very basic INSTRUCTION_PROMPT")



    if model_name=='intern':
        from agent import InternVLM
        agent = InternVLM()
    elif model_name=='qwen':
        from agent import QwenVLM
        agent = QwenVLM()
    elif model_name=='qwen-32':
        from agent import QwenVLM
        agent = QwenVLM(model_path='/home/homesOnMaster/zzhao/weights/Qwen3-VL-32B-Instruct')
    elif model_name=='gemma':
        from agent import GemmaVLM
        agent = GemmaVLM()
    elif model_name=='medgemma-27':
        from agent import GemmaVLM
        agent = GemmaVLM(model_path='/home/homesOnMaster/zzhao/weights/medgemma-27b-it')
    elif model_name=='medgemma_15':
        from agent import GemmaVLM
        agent = GemmaVLM(model_path='/home/homesOnMaster/zzhao/weights/medgemma-1.5-4b-it')
    elif model_name=='glm':
        from agent import VisGLM
        agent = VisGLM(model_path=os.path.join(weights_prefix, 'GLM-4.6V-Flash'))
    elif model_name=='medvlm':
        from agent import MedVLM_R1
        agent = MedVLM_R1()
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    # pdb.set_trace()

    info=pd.read_csv(os.path.join(data_prefix,csv_path))


    # pdb.set_trace()
    results=[]
    for i, row in tqdm(info.iterrows(), total=len(info), desc="Processing samples"):
        subject_id=row['subject_id']
        study_id=row['study_id']
        edema_label=row['Edema']
        pneumonia_label=row['Pneumonia']
        img_path_list=ast.literal_eval(row['img_paths'])
        img_path_list=[os.path.join(data_prefix,path) for path in img_path_list]
        messages_edema = {
        'image': img_path_list,
        'text': ['Please analyze the provided medical images. They belong to the same subject, so you need to summarize these several views and find evidence supporting the existence of edema.'],
        'instruction': EDEMA_AGENT_PROMPT
        }
        messages_pneumonia = {
        'image': img_path_list,
        'text': ['Please analyze the provided medical images. They belong to the same subject, so you need to summarize these several views and find evidence supporting the existence of pneumonia.'],
        'instruction': PNEUMONIA_AGENT_PROMPT
        }
        # pdb.set_trace()
        edema_evidence=agent.chat(messages_edema)
        # pdb.set_trace()
        pneumonia_evidence=agent.chat(messages_pneumonia)
        messages_judge = {
        'image': [],
        'text': [f"Agent-Edema's evidence summary {edema_evidence}",f"Agent-Pneumonia's evidence summary {pneumonia_evidence}"],
        'instruction': JUDGE_IMAGE_TEXT_AGENT_PROMPT
        }
        # pdb.set_trace()
        response=agent.chat(messages_judge)
        # pdb.set_trace()
        # pdb.set_trace()
        results.append({
            'subject_id': subject_id,
            'study_id': study_id,
            'Edema': edema_label,
            'Pneumonia': pneumonia_label,
            'response': response,
            'edema_evidence':edema_evidence,
            'pneumonia evidence': pneumonia_evidence

        })
        if i%50==0:
            # -------------------------------
            # Save intermediate results to CSV
            # -------------------------------
            df_out = pd.DataFrame(results)
            df_out.to_csv(os.path.join(data_prefix,output_path), index=False)

    # -------------------------------
    # Save results to CSV
    # -------------------------------
    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(data_prefix,output_path), index=False)
