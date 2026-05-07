import re
import csv
from pathlib import Path

class PredictResult:
    def __init__(self, resource_file, model, result_txt):
        self.resource_file = resource_file
        self.model = model
        self.result_txt = result_txt
        self.result = None
        self.trial = 'unknown' # 1: 结果正确，0：结果错误, -1：无法判断

    def parse_result(self):
        # 定义正则表达式模式，匹配predict_result(...)中的内容
        pattern = r'predict_result\((.*?)\)'

        # 查找匹配
        match = re.search(pattern, self.result_txt)

        if match:
            self.result = match.group(1)

        if self.result is not None:
            if self.result == self.resource_file.correct_result:
                self.trial = 'correct'
            else:
                self.trial = 'wrong'

    def save_to_csv(self, csv_path):
        # 确定文件是否存在
        file_exists = Path(csv_path).exists()
        fieldnames = [
            'group',
            'image',
            'model',
            'result',
            'trial',
            'result_txt',
        ]
        # 根据append参数确定文件打开模式
        mode = 'a' if file_exists else 'w'
        with open(csv_path, mode, newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_NONNUMERIC
            )

            # 如果是新文件或覆盖模式，写入表头
            if not file_exists:
                writer.writeheader()

            # 写入当前实例数据
            writer.writerow({
                'group': self.resource_file.group,
                'image': self.resource_file.filename,
                'model': self.model,
                'result': self.result,
                'trial': self.trial,
                'result_txt': self.result_txt
            })