from google import genai
from google.genai import types


import os
from openai import OpenAI
from PROMPT import INSTRUCTION_PROMPT,EDEMA_AGENT_PROMPT_v2,PNEUMONIA_AGENT_PROMPT_v2,JUDGE_IMAGE_TEXT_AGENT_PROMPT
import pandas as pd
import ast
import argparse
import os
import time
import pdb
from tqdm import tqdm
import base64
import httpx

import anthropic

from google.genai import types


# Function to create a file with the Files API
def create_file(client, file_path):
    with open(file_path, 'rb') as f:
        image_bytes = f.read()

    file_data=types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/jpeg',
      ),

  
    return file_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default='mimic-cxr-edema-pneumonia-dcm_filtered.csv',
                        help="Path to the input CSV file")
    parser.add_argument("--output_path", type=str, default='./cxr_gemini_3_flash_care.csv')
    model_name="gemini-3-flash-preview"
    args = parser.parse_args()
    data_prefix='./data/mimic-cxr-jpg-512/'
    csv_path = args.csv_path
    output_path = args.output_path


    # pdb.set_trace()

    info=pd.read_csv(os.path.join(data_prefix,csv_path))

    results=[]
    client = genai.Client(api_key = "xxxxxxxxxxxxxxx")

    MAX_RETRY = 5

    for iter_idx, row in tqdm(info.iterrows(), total=len(info), desc="Processing samples"):
        subject_id = row['subject_id']
        study_id = row['study_id']
        edema_label = row['Edema']
        pneumonia_label = row['Pneumonia']

        success = False

        for attempt in range(MAX_RETRY):
            try:
                # -------------------------------
                # -------------------------------
                img_path_list = ast.literal_eval(row['img_paths'])
                img_path_list = [os.path.join(data_prefix, p) for p in img_path_list]

                edema_content_list = [
                    EDEMA_AGENT_PROMPT_v2
                ]
                pneumonia_content_list = [
                    PNEUMONIA_AGENT_PROMPT_v2   
                ]
                judge_content_list = [
                    JUDGE_IMAGE_TEXT_AGENT_PROMPT   
                ]
                
                # -------------------------------
                # -------------------------------
                for elem in img_path_list:
                    file_data=create_file(client,elem)
                    edema_content_list.append(file_data)
                    pneumonia_content_list.append(file_data)
                    judge_content_list.append(file_data)


                # -------------------------------
                # -------------------------------
                response_edema = client.models.generate_content(
                    model=model_name,
                    contents=edema_content_list,
                )
                judge_content_list.append(response_edema.text)
                response_pneumonia = client.models.generate_content(
                    model=model_name,
                    contents=pneumonia_content_list,
                )
                judge_content_list.append(response_pneumonia.text)
                response_judge = client.models.generate_content(
                    model=model_name,
                    contents=judge_content_list,
                )
                #     input=[{
                #         "role": "user",
                #         "content": content_list
                #     }],
                # )
                
                # response_text = response.output_text

                success = True
                break   

            except Exception as e:
                print(f"[Retry {attempt+1}/{MAX_RETRY}] subject_id={subject_id} | {e}")
                time.sleep(2)  

        if not success:
            print(f"[FAILED] subject_id={subject_id}, skipped after {MAX_RETRY} retries.")
            continue

        results.append({
            'subject_id': subject_id,
            'study_id': study_id,
            'Edema': edema_label,
            'Pneumonia': pneumonia_label,
            'response_edema': response_edema.text,
            'response_pneumonia': response_pneumonia.text,
            'response': response_judge.text
        })

        if iter_idx % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

    
    # -------------------------------
    # Save results to CSV
    # -------------------------------
    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(data_prefix,output_path), index=False)


