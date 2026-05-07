from openai import OpenAI
import base64
import os
import pandas as pd
from datetime import datetime
import config
import resources

AI_MODELS = [
    "gpt-5",
    "claude-sonnet-4-20250514",
    "gemini-2.5-pro",
    "doubao-seed-1-6-250615",
    "Qwen/Qwen2.5-VL-72B-Instruct"
]

class AiClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://aihubmix.com/v1"
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url
        )

    def predict_with_img(self, img_path, model_name, prompt):
        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        try:
            chat_completion = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            response = chat_completion.choices[0].message.content
            return response
        except Exception as e:
            print(f"处理图片 {img_path} 时出错（模型: {img_path}）: {str(e)}")
            return None

# 图片文
# if __name__ == "__main__":
#     # 读取并解析YAML文件
#     yaml_data = config.load_yaml_file('../conf/config.test.yaml')
#     if not yaml_data:
#         print("failed to load config.test.yaml")
#         exit(-1)
#
#     api_key = yaml_data.get('api_key')
#     resource_path = yaml_data.get('resource_path')
#
#     root_dir = os.getcwd() + '/../' + resource_path
#     imgs = resources.collect_resources(root_dir)
#
#     client = AiClient(api_key)
#     for img in imgs:
#         for model in AI_MODELS:
#             print(f">>>>> 处理图片: {img.fullpath}")
#             print(f">>>>> 正在使用模型: {model}")
#
#             rsp = client.predict_with_img(img.fullpath, model, img.prompt)
#             if rsp is not None:
#                 print(f"predict success, result is: {rsp}")
#             else:
#                 print(f"predict failed")
