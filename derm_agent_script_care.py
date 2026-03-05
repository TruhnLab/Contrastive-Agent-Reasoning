
from PROMPT import INSTRUCTION_PROMPT,KNOWLEDGE_AGENT_PROMPT,INSTRUCTION_DERMA_PROMPT,ATYPICAL_NEVUS_AGENT_PROMPT,MELANOMA_AGENT_PROMPT,DERM_JUDGE_IMAGE_TEXT_AGENT_PROMPT
import pandas as pd
import ast
import argparse
import os
import pdb
from tqdm import tqdm



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default='derm7pt_filtered_meta.csv',
                        help="Path to the input CSV file")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["phi", "intern", "qwen", "deepseek","gemma","qwen-32","medgemma-27","medvlm","medgemma_15","glm"],
                        help="Which VLM to use")
    parser.add_argument("--output_path", type=str, default='./medgemma_1.5_4b.csv')

    args = parser.parse_args()
    data_prefix='/home/homesOnMaster/zzhao/data/derm7pt/'
    weights_prefix='/home/homesOnMaster/zzhao/weights/'
    csv_path = args.csv_path
    output_path = args.output_path
    model_name = args.model_name

    Instruction_prompt=INSTRUCTION_DERMA_PROMPT
    print("Using the very basic INSTRUCTION_PROMPT")



    
    # if model_name=='phi':
    #     from agent import PhiVLM
    #     agent = PhiVLM()
    if model_name=='intern':
        from agent import InternVLM
        agent = InternVLM(os.path.join(weights_prefix, 'InternVL3-14B'))
    elif model_name=='qwen':
        from agent import QwenVLM
        agent = QwenVLM(os.path.join(weights_prefix, 'Qwen3-VL-8B-Instruct'))
    elif model_name=='qwen-32':
        from agent import QwenVLM
        agent = QwenVLM(model_path=os.path.join(weights_prefix, 'Qwen3-VL-32B-Instruct'))
    elif model_name=='gemma':
        from agent import GemmaVLM
        agent = GemmaVLM(model_path=os.path.join(weights_prefix, 'gemma-3-12b-it'))
    elif model_name=='medgemma-27':
        from agent import GemmaVLM
        agent = GemmaVLM(model_path=os.path.join(weights_prefix, 'medgemma-27b-it'))
    elif model_name=='medgemma_15':
        from agent import GemmaVLM
        agent = GemmaVLM(model_path=os.path.join(weights_prefix, 'medgemma-1.5-4b-it'))
    # elif model_name=='deepseek':
    #     from agent import DeepSeekVLM
    #     agent = DeepSeekVLM()
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

    results=[]
    for i, row in tqdm(info.iterrows(), total=len(info), desc="Processing samples"):
        case_id = row['case_num']
        melanoma_label = row['melanoma']
        gt_label=row['diagnosis']
        atypical_nevus_label = row['atypical_nevus']
        img_path_list = [row['derm']]
        img_path_list = [os.path.join(data_prefix,"images", p) for p in img_path_list]

        atypical_nevus_messages = {
        'image': img_path_list,
        'text':"",
        'instruction': ATYPICAL_NEVUS_AGENT_PROMPT
        }
        melanoma_messages = {
        'image': img_path_list,
        'text':"",
        'instruction': MELANOMA_AGENT_PROMPT
        }
        # pdb.set_trace()
        response_atypical_nevus = agent.chat(atypical_nevus_messages)
        if not isinstance(response_atypical_nevus, str):
            response_atypical_nevus = response_atypical_nevus.text
        
        response_melanoma = agent.chat(melanoma_messages)
        if not isinstance(response_melanoma, str):
            response_melanoma = response_melanoma.text
        judge_messages = {
        'image': img_path_list,
        'text': f"{response_atypical_nevus}\n{response_melanoma}",
        'instruction': DERM_JUDGE_IMAGE_TEXT_AGENT_PROMPT
        }
        response_judge = agent.chat(judge_messages)
        if not isinstance(response_judge, str):
            response_judge = response_judge.text
        # pdb.set_trace()
        results.append({
            'case_id': case_id,
            'Melanoma': melanoma_label,
            'Atypical_Nevus': atypical_nevus_label,
            "fine_grained_gt": gt_label,
            'response_atypical_nevus': response_atypical_nevus,
            'response_melanoma': response_melanoma,
            'response': response_judge
        })
        if i%10==0:
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