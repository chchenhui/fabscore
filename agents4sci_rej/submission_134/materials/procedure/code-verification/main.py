import csv
import os
from datetime import datetime

import config
import resources
import ai_client
import predict_result

if __name__ == "__main__":
    # 1. 初始化
    result_path = r'/Users/chenrui/Desktop/zhang/ai_benchmark-main/result.csv'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    yaml_data = config.load_yaml_file('../conf/config.test.yaml')
    if not yaml_data:
        print("failed to load config.test.yaml")
        exit(-1)

    api_key      = yaml_data.get('api_key')
    resource_dir = os.path.join(os.getcwd(), '..', yaml_data.get('resource_path'))
    imgs         = resources.collect_resources(resource_dir)
    client       = ai_client.AiClient(api_key)

    # 2. 时间戳，用于文件名
    ts = datetime.now().strftime("%Y%m%d%H%M")

    # 3. 创建每个模型的 csv（先写表头）
    model_files = {}
    for model in ai_client.AI_MODELS:
        if "Qwen" in model:
            f_name = f"Qwen.csv"
        else:
            f_name = f"{model}.csv"
        model_files[model] = open(f_name, 'w', newline='', encoding='utf-8')
        writer = csv.writer(model_files[model])
        # 表头：图片名 + 10 次答案
        writer.writerow(['pictures'] + [f"{model}_{i+1}" for i in range(10)])

    # 4. 创建跨模型的总表
    with open(result_path, 'w', newline='', encoding='utf-8') as re_csv:
        writer = csv.writer(re_csv)
        header = ['pictures'] + ai_client.AI_MODELS + ['correct_ratio']
        writer.writerow(header)

        # 5. 主循环
        for img in imgs:   # 仅演示前 5 张
            img_name = os.path.basename(img.fullpath)
            row_total = [img_name]
            total_correct = 0          # 用来算综合正确率
            total_trials  = 0

            for model in ai_client.AI_MODELS:
                model_correct = 0
                model_answers = [img_name]

                for run in range(10):
                    print(f">>>>> 处理图片: {img.fullpath} | 模型: {model} 第 {run+1}/10 次")
                    rsp = client.predict_with_img(img.fullpath, model, img.prompt)
                    if rsp is None:
                        ans = 'fail'
                    else:
                        r = predict_result.PredictResult(img, model, rsp)
                        r.parse_result()
                        r.save_to_csv(f"../result.{ts}.csv")   # 保持原单条记录
                        ans = r.trial
                        if ans == 'correct':
                            model_correct += 1
                    model_answers.append(ans)

                # 写入该模型的专属 csv
                csv.writer(model_files[model]).writerow(model_answers)

                # 计算该模型 10 次正确率
                model_ratio = model_correct / 10.0
                total_correct += model_correct
                total_trials  += 10
                row_total.append(f"{model_ratio:.2f}")

            # 综合正确率
            overall_ratio = total_correct / total_trials if total_trials else 0.0
            row_total.append(f"{overall_ratio:.2f}")
            writer.writerow(row_total)

    # 6. 关闭所有模型文件
    for f in model_files.values():
        f.close()

    print("全部完成！")