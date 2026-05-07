import os
import config

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

class ResourceFile:
    def __init__(self, fullpath, group, filename, prompt, correct_result):
        self.fullpath = fullpath
        self.group = group
        self.filename = filename
        self.prompt = prompt
        self.correct_result = correct_result


def read_trial_file(filename):
    result = {}
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行首尾的空白字符（包括换行符）
                line = line.strip()
                # 跳过空行
                if not line:
                    continue
                # 按空格分割列（假设列之间用空格分隔）
                parts = line.split()
                # 确保每行有两列
                if len(parts) >= 2:
                    key = parts[0]
                    value = parts[1]
                    result[key] = value
                else:
                    print(f"警告：跳过格式不正确的行 - {line}")
                    exit(-1)
        return result
    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 未找到")
        exit(-1)
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        exit(-1)


def collect_resources(root_dir):
    subdirectories = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath != root_dir:
            group_name = os.path.relpath(dirpath, root_dir)
            subdirectories.append((dirpath, group_name))

    resource_files = []
    for dirpath, group_name in subdirectories:
        prompt_path = os.path.join(dirpath, 'prompt.txt')
        with open(prompt_path, 'r', encoding='utf-8') as file:
            # 读取文件内容
            prompt = file.read()
        trial_path = os.path.join(dirpath, 'trial.txt')
        trials = read_trial_file(trial_path)
        # 遍历当前子目录中的所有文件
        for filename in os.listdir(dirpath):
            file_path = os.path.join(dirpath, filename)
            # 检查是否为文件且是图片格式
            if os.path.isfile(file_path) and filename.lower().endswith(IMAGE_EXTENSIONS):
                correct_result = trials.get(filename)
                if correct_result is None:
                    print(f"错误：文件 '{filename}' 不在trial.txt中，完整路径 {file_path}")
                    exit(-1)
                else:
                    resource_files.append(ResourceFile(file_path, group_name, filename, prompt, correct_result))
    return resource_files

# if __name__ == '__main__':
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
#
#     for resource in collect_resources(root_dir):
#         print(f"{resource.group} {resource.filename} {resource.correct_result} {resource.fullpath}")
