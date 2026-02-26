from google import genai
from google.genai import types
import os
from openai import OpenAI
from PROMPT import INSTRUCTION_DERMA_PROMPT,ATYPICAL_NEVUS_AGENT_PROMPT,MELANOMA_AGENT_PROMPT,DERM_JUDGE_IMAGE_TEXT_AGENT_PROMPT,DERM_JUDGE_TEXT_AGENT_PROMPT
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
    parser.add_argument("--output_path", type=str, default='./derm_gemini_care_text_only.csv')
    model_name="gemini-3-flash-preview"
    # model_name="gemini-3-pro-preview"

    args = parser.parse_args()
    data_prefix=os.path.expanduser('~/Documents/data/derm7pt/')
    csv_path = args.csv_path
    output_path = args.output_path

    # pdb.set_trace()

    info=pd.read_csv(os.path.join(data_prefix,csv_path))

    results=[]
    client = genai.Client(api_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


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
                # 解析图像路径
                # -------------------------------
                img_path_list = [row['derm']]
                img_path_list = [os.path.join(data_prefix,"images", p) for p in img_path_list]

                atypical_nevus_content_list = [
                    ATYPICAL_NEVUS_AGENT_PROMPT
                ]
                melanoma_content_list = [
                    MELANOMA_AGENT_PROMPT   
                ]
                judge_content_list = [
                    DERM_JUDGE_TEXT_AGENT_PROMPT   
                ]
                
                # -------------------------------
                # 上传图片（网络最容易抖）
                # -------------------------------
                for elem in img_path_list:
                    file_data=create_file(client,elem)
                    atypical_nevus_content_list.append(file_data)
                    melanoma_content_list.append(file_data)
                    # image_data=base64.standard_b64encode(httpx.get(elem).content).decode("utf-8")
                    # media_type="image/jpeg"
                    # content_list.append({
                    #     "type": "image",
                    #     "source":{
                    #         "type":"base64",
                    #         "media_type": media_type,
                    #         "data":image_data,
                    #     }
                        
                    # })


                # -------------------------------
                # 调用模型
                # -------------------------------
                response_atypical_nevus = client.models.generate_content(
                    model=model_name,
                    contents=atypical_nevus_content_list,
                )
                judge_content_list.append(response_atypical_nevus.text)

                response_melanoma = client.models.generate_content(
                    model=model_name,
                    contents=melanoma_content_list,
                )
                judge_content_list.append(response_melanoma.text)

                response_judge = client.models.generate_content(
                    model=model_name,
                    contents=judge_content_list,
                )
                # response = client.responses.create(
                #     model="claude-sonnet-4-5",
                #     input=[{
                #         "role": "user",
                #         "content": content_list
                #     }],
                # )
                
                # response_text = response.output_text

                success = True
                break   # ✅ 成功就跳出 retry 循环

            except Exception as e:
                print(f"[Retry {attempt+1}/{MAX_RETRY}] case_id={case_id} | {e}")
                time.sleep(2)  # 简单等待，防止立刻再炸

        # -------------------------------
        # 5 次都失败才放弃这个 subject
        # -------------------------------
        if not success:
            print(f"[FAILED] case_id={case_id}, skipped after {MAX_RETRY} retries.")
            continue

        # -------------------------------
        # 成功结果写入
        # -------------------------------
        results.append({
            'case_id': case_id,
            'Melanoma': melanoma_label,
            'Atypical_Nevus': atypical_nevus_label,
            "fine_grained_gt": gt_label,
            "response_atypical_nevus": response_atypical_nevus.text,
            "response_melanoma": response_melanoma.text,
            'response': response_judge.text
        })

        if iter_idx % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

    
    # -------------------------------
    # Save results to CSV
    # -------------------------------
    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(data_prefix,output_path), index=False)


    # print("Response from VLM:")
    # print(response)