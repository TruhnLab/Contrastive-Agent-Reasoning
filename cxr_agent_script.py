
from PROMPT import INSTRUCTION_PROMPT,KNOWLEDGE_AGENT_PROMPT
import pandas as pd
import ast
import argparse
import os
import pdb
from tqdm import tqdm



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default='mimic-cxr-edema-pneumonia-dcm_filtered.csv',
                        help="Path to the input CSV file")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["phi", "intern", "qwen", "deepseek","gemma","qwen-32","medgemma-27","medvlm","medgemma_15","glm"],
                        help="Which VLM to use")
    parser.add_argument("--output_path", type=str, default='./medgemma_1.5_4b.csv')

    args = parser.parse_args()
    data_prefix='/home/homesOnMaster/zzhao/data/mimic-cxr-jpg-512/'
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

    print(f"Samples after filtering: {len(info)}")
    results=[]
    for i, row in tqdm(info.iterrows(), total=len(info), desc="Processing samples"):
        subject_id=row['subject_id']
        study_id=row['study_id']
        edema_label=row['Edema']
        pneumonia_label=row['Pneumonia']
        img_path_list=ast.literal_eval(row['img_paths'])
        img_path_list=[os.path.join(data_prefix,path) for path in img_path_list]
        messages = {
        'image': img_path_list,
        'text': ['Please analyze the provided medical images. They belong to the same subject, so you need to summarize these several views to discriminate whether this subject is contracted with Edema or Pneumonia.'],
        'instruction': Instruction_prompt
        }
        # pdb.set_trace()
        response = agent.chat(messages)
        # pdb.set_trace()
        results.append({
            'subject_id': subject_id,
            'study_id': study_id,
            'Edema': edema_label,
            'Pneumonia': pneumonia_label,
            'response': response
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


    # print("Response from VLM:")
    # print(response)