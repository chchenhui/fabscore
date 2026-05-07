import yaml
import os


def load_yaml_file(file_path):
    """
    读取并解析YAML文件

    参数:
        file_path (str): YAML文件的路径

    返回:
        dict: 解析后的YAML数据，如果出错则返回None
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return None

    try:
        # 打开并解析YAML文件
        with open(file_path, 'r', encoding='utf-8') as file:
            # safe_load 方法更安全，避免执行恶意代码
            data = yaml.safe_load(file)
            return data
    except yaml.YAMLError as e:
        print(f"YAML解析错误: {e}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

    return None


def save_to_yaml(data, file_path):
    """
    将数据保存为YAML文件

    参数:
        data (dict): 要保存的数据
        file_path (str): 保存的文件路径
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            # default_flow_style=False 使输出更易读
            yaml.dump(data, file, allow_unicode=True, default_flow_style=False)
        print(f"数据已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存YAML文件时发生错误: {e}")


# if __name__ == "__main__":
#     # 读取并解析YAML文件
#     yaml_data = load_yaml_file('../conf/config.yaml')
#
#     if yaml_data:
#         print("\n解析后的YAML数据:")
#         print(yaml_data)
#
#         # 访问特定数据
#         print("\n访问特定数据:")
#         print(f"配置名称: {yaml_data.get('api_key')}")
#         print(f"配置名称: {yaml_data.get('resource_path')}")
