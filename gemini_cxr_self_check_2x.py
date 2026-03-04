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
    parser.add_argument("--output_path", type=str, default='./cxr_gemini_3_flash_cxr_self-check-2x.csv')
    model_name="gemini-3-flash-preview"
    args = parser.parse_args()
    data_prefix='./data/mimic-cxr-jpg-512/'
    csv_path = args.csv_path
    output_path = args.output_path


    # pdb.set_trace()

    info=pd.read_csv(os.path.join(data_prefix,csv_path))

    results=[]
    client = genai.Client(api_key = "xxxxxxxxxxxxxx")

    MAX_RETRY = 5

    for iter_idx, row in tqdm(info.iterrows(), total=len(info), desc="Processing samples"):
        # 1490-1740
        if iter_idx<=1180:
            continue
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

                content_list = [
                    INSTRUCTION_PROMPT,
                ]
                
                # -------------------------------
                # -------------------------------
                for elem in img_path_list:
                    file_data=create_file(client,elem)
                    content_list.append(file_data)

                response_init = client.models.generate_content(
                    model=model_name,
                    contents=content_list,
                )
                content_list.append(f"Here is your previous diagnosis: {response_init.text} \n\nRe-examine the IMAGE carefully.\n- Verify each claimed visual finding.\n- Identify any over-interpretation or unsupported claim.\n- Revise the diagnosis if needed.\n Your output should strictly follow the orignial format requirement.")            
                revised_response = client.models.generate_content(
                    model=model_name,
                    contents=content_list,
                ).text



                #     input=[{
                #         "role": "user",
                #         "content": content_list
                #     }],
                # )
                
                # response_text = response.output_text

                success = True
                break   # 

            except Exception as e:
                print(f"[Retry {attempt+1}/{MAX_RETRY}] subject_id={subject_id} | {e}")
                time.sleep(2)  #

        # -------------------------------
        # -------------------------------
        if not success:
            print(f"[FAILED] subject_id={subject_id}, skipped after {MAX_RETRY} retries.")
            continue

        # -------------------------------
        # -------------------------------
        results.append({
            'subject_id': subject_id,
            'study_id': study_id,
            'Edema': edema_label,
            'Pneumonia': pneumonia_label,
            'response_init': response_init.text,
            'response': revised_response
        })

        if iter_idx % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

    
    # -------------------------------
    # Save results to CSV
    # -------------------------------
    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(data_prefix,output_path), index=False)

