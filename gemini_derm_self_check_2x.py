from google import genai
from google.genai import types
import os
from openai import OpenAI
from PROMPT import INSTRUCTION_DERMA_PROMPT
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
    parser.add_argument("--csv_path", type=str, default='derm7pt_filtered_meta.csv',
                        help="Path to the input CSV file")
    parser.add_argument("--output_path", type=str, default='./derm_gemini_3_flash_self-check-2x.csv')
    model_name="gemini-3-flash-preview"
    # model_name="gemini-3-pro-preview"

    args = parser.parse_args()
    data_prefix=os.path.expanduser('./data/derm7pt/')
    csv_path = args.csv_path
    output_path = args.output_path

    # pdb.set_trace()

    info=pd.read_csv(os.path.join(data_prefix,csv_path))

    results=[]
    client = genai.Client(api_key = "xxxxxxxxxxxxxxx")


    MAX_RETRY = 5

    for iter_idx, row in tqdm(info.iterrows(), total=len(info), desc="Processing samples"):
        case_id = row['case_num']
        melanoma_label = row['melanoma']
        gt_label=row['diagnosis']
        atypical_nevus_label = row['atypical_nevus']

        success = False

        for attempt in range(MAX_RETRY):
            try:
                # -------------------------------
                # -------------------------------
                img_path_list = [row['derm']]
                img_path_list = [os.path.join(data_prefix,"images", p) for p in img_path_list]

                content_list = [
                    INSTRUCTION_DERMA_PROMPT,
                ]
                
                # -------------------------------
                # -------------------------------
                for elem in img_path_list:
                    file_data=create_file(client,elem)
                    content_list.append(file_data)



                # -------------------------------
                # -------------------------------
                response_init = client.models.generate_content(
                    model=model_name,
                    contents=content_list,
                )
                content_list.append(f"Here is your previous diagnosis: {response_init.text} \n\nRe-examine the IMAGE carefully.\n- Verify each claimed visual finding.\n- Identify any over-interpretation or unsupported claim.\n- Revise the diagnosis if needed.\n Your output should strictly follow the orignial format requirement.")            
                revised_response = client.models.generate_content(
                    model=model_name,
                    contents=content_list,
                ).text
                success = True
                break   # 

            except Exception as e:
                print(f"[Retry {attempt+1}/{MAX_RETRY}] case_id={case_id} | {e}")
                time.sleep(2)  # 

        # -------------------------------
        # -------------------------------
        if not success:
            print(f"[FAILED] case_id={case_id}, skipped after {MAX_RETRY} retries.")
            continue

        # -------------------------------
        # -------------------------------
        results.append({
            'case_id': case_id,
            'Melanoma': melanoma_label,
            'Atypical_Nevus': atypical_nevus_label,
            "fine_grained_gt": gt_label,
            'response_init': response_init.text,
            'response': revised_response
        })

        if iter_idx % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

    
    # -------------------------------
    # -------------------------------
    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(data_prefix,output_path), index=False)


