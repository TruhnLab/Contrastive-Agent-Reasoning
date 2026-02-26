
from PROMPT import INSTRUCTION_PROMPT,KNOWLEDGE_AGENT_PROMPT,INSTRUCTION_DERMA_PROMPT
import pandas as pd
import ast
import argparse
import os
import pdb
from tqdm import tqdm

def extract_decision(response):
    response_text = str(response)

    try:
        response_text = response_text.encode().decode("unicode_escape")
    except Exception:
        pass

    try:
        match = re.search(
            r'<<<FINAL_DECISION>>>.*?Finding:\s*([^\n]+)',
            response_text,
            re.DOTALL,
        )
        if match:
            finding = match.group(1).strip()
            if finding.lower() == "edema or pneumonia" or finding.lower() == "uncertain":
                return "uncertain"

            return finding
    except Exception:
        pass

    return "uncertain"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default='derm7pt_filtered_meta.csv',
                        help="Path to the input CSV file")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["phi", "intern", "qwen", "deepseek","gemma","qwen-32","medgemma-27","medvlm","medgemma_15","glm"],
                        help="Which VLM to use")
    parser.add_argument("--output_path", type=str, default='./medgemma_1.5_4b.csv')

    args = parser.parse_args()
    # data_prefix='/home/homesOnMaster/zzhao/data/derm7pt/'
    # weights_prefix='/home/homesOnMaster/zzhao/weights/'
    data_prefix='/hpcwork/qdt11020/data/derm7pt/'
    weights_prefix='/hpcwork/qdt11020/weights/'
    csv_path = args.csv_path
    output_path = args.output_path
    model_name = args.model_name

    Instruction_prompt=INSTRUCTION_PROMPT
    print("Using the very basic INSTRUCTION_PROMPT")



    
    if model_name=='phi':
        from agent import PhiVLM
        agent = PhiVLM()
    elif model_name=='intern':
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
    elif model_name=='deepseek':
        from agent import DeepSeekVLM
        agent = DeepSeekVLM()
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

    # the_chosen=info.sample()
    # img_path_list=ast.literal_eval(the_chosen['img_paths'].iloc[0])
    # the_chosen=info.iloc[3]
    # img_path_list=ast.literal_eval(the_chosen['img_paths'])
    # img_path_list=[os.path.join(data_prefix,path) for path in img_path_list]
    results=[]
    for i, row in tqdm(info.iterrows(), total=len(info), desc="Processing samples"):
        case_id = row['case_num']
        melanoma_label = row['melanoma']
        gt_label=row['diagnosis']
        atypical_nevus_label = row['atypical_nevus']
        img_path_list = [row['derm']]
        img_path_list = [os.path.join(data_prefix,"images", p) for p in img_path_list]
        messages = {
        'image': img_path_list,
        'text':"atypical nevus or melanoma?",
        'instruction': INSTRUCTION_DERMA_PROMPT
        }
        # pdb.set_trace()
        response = agent.chat(messages)
        # pdb.set_trace()
        results.append({
            'case_id': case_id,
            'Melanoma': melanoma_label,
            'Atypical_Nevus': atypical_nevus_label,
            "fine_grained_gt": gt_label,
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