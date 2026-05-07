#!/usr/bin/env python3
"""
分析LLM回答的脚本
读取LLM的JSON结果文件，使用qwen AI分析LLM对原始内容的判断，返回五分类结果
支持分析deepseek-r1和qwen3等多种模型的回答
"""
import os
import json
import time
import asyncio
import aiohttp
from typing import List, Dict, Any
import logging

# 设置日志 - 增加详细日志输出
logging.basicConfig(
    level=logging.INFO,  # 可以改为DEBUG查看更详细信息
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler('llm_analysis.log', encoding='utf-8')  # 文件输出
    ]
)
logger = logging.getLogger(__name__)

class LLMResponseAnalyzer:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.semaphore = asyncio.Semaphore(480)  # 480个并发
        self.last_request_time = 0
        self.request_interval = 1.0 / 480  # 每个请求间隔约0.002秒
    
    async def analyze_llm_response(self, session: aiohttp.ClientSession, llm_response: str, original_text: str, record_id: str) -> tuple:
        """分析单个LLM回答"""
        prompt = f"""请分析以下AI回答，提取它对原始内容的核心判断。

原始微博内容：{original_text}

AI的分析回答：{llm_response}

请判断AI认为原始内容是：
- 1（错误）：AI明确认为原始内容是错误的、不准确的、有害的
- 2（可疑错误）：AI倾向于认为原始内容可能有问题，但不完全确定
- 3（模糊）：AI认为原始内容模糊不清、需要更多信息、或持完全中性态度
- 4（可疑正确）：AI倾向于认为原始内容可能是正确的，但有一些保留意见
- 5（正确）：AI明确认为原始内容是正确的、准确的、有依据的

请仔细阅读AI回答的整体倾向和结论，只回答一个数字：1、2、3、4或5"""

        async with self.semaphore:  # 限制并发数量
            # 控制请求频率：每秒480个
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.request_interval:
                await asyncio.sleep(self.request_interval - time_since_last)
            self.last_request_time = time.time()
            
            logger.debug(f"开始分析记录 {record_id}")
            
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "你是一个专业的医疗健康内容评估专家，擅长分析AI模型对健康信息的判断倾向。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 10
                }
                
                start_time = time.time()
                async with session.post(f"{self.base_url}/chat/completions", 
                                      headers=headers, 
                                      json=data,
                                      timeout=aiohttp.ClientTimeout(total=60)) as response:  # 增加超时时间到60秒
                    
                    request_time = time.time() - start_time
                    
                    if response.status == 200:
                        result_data = await response.json()
                        result = result_data["choices"][0]["message"]["content"].strip()
                        
                        # 确保返回值是1、2、3、4、5之一
                        if result in ['1', '2', '3', '4', '5']:
                            logger.debug(f"记录 {record_id} 分析成功: {result} (耗时: {request_time:.2f}s)")
                            return record_id, result, None
                        # 容错处理，提取第一个有效数字
                        for char in result:
                            if char in ['1', '2', '3', '4', '5']:
                                logger.debug(f"记录 {record_id} 分析成功(容错): {char} (耗时: {request_time:.2f}s)")
                                return record_id, char, None
                        
                        logger.warning(f"记录 {record_id} 意外的返回值: {result} (耗时: {request_time:.2f}s)")
                        return record_id, '3', None
                    else:
                        error_text = await response.text()
                        error_msg = f"HTTP {response.status}: {error_text}"
                        logger.error(f"记录 {record_id} API调用失败: {error_msg} (耗时: {request_time:.2f}s)")
                        return record_id, None, error_msg
                        
            except Exception as e:
                logger.error(f"记录 {record_id} 调用异常: {str(e)}")
                return record_id, None, str(e)
    
    def read_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """读取JSON文件并返回数据列表"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功读取 {file_path}，共 {len(data)} 条记录")
            return data
        except Exception as e:
            logger.error(f"读取文件 {file_path} 失败: {e}")
            return []
    
    async def process_single_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """处理单个JSON文件"""
        logger.info(f"开始处理文件: {file_path}")
        data = self.read_json_file(file_path)
        
        # 创建所有任务
        tasks = []
        task_records = {}  # 用于存储记录和任务的映射
        
        async with aiohttp.ClientSession() as session:
            for i, record in enumerate(data):
                response_text = record.get('response', '')
                original_text = record.get('text', '')
                record_id = f"{i}"
                
                if not response_text:
                    logger.warning(f"文件 {file_path} 第 {i+1} 条记录没有response内容")
                    # 直接设置默认值，不需要API调用
                    task_records[record_id] = (record, '3')
                else:
                    # 创建API调用任务
                    task = self.analyze_llm_response(session, response_text, original_text, record_id)
                    tasks.append(task)
                    task_records[record_id] = (record, None)  # None表示需要等待API结果
            
            logger.info(f"创建了 {len(tasks)} 个API调用任务")
            
            # 并发执行所有任务
            if tasks:
                logger.info("开始并发执行所有API任务...")
                start_time = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                total_time = time.time() - start_time
                
                logger.info(f"所有API任务执行完成，总耗时: {total_time:.2f}s")
                
                # 处理结果
                success_count = 0
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"任务执行出错: {result}")
                        continue
                    
                    record_id, classification, error = result
                    if error:
                        logger.warning(f"记录 {record_id} 分析失败: {error}")
                        # 标记为失败，稍后重试
                        task_records[record_id] = (task_records[record_id][0], None)
                    else:
                        # 成功，更新结果
                        task_records[record_id] = (task_records[record_id][0], classification)
                        success_count += 1
                
                logger.info(f"首轮执行结果: 成功 {success_count} 个，失败 {len(tasks) - success_count} 个")
            
            # 重试失败的任务，最多50次
            retry_count = 0
            max_retries = 10
            
            while retry_count < max_retries:
                # 找出需要重试的任务
                retry_tasks = []
                retry_record_ids = []
                
                for record_id, (record, classification) in task_records.items():
                    if classification is None:  # 需要重试
                        response_text = record.get('response', '')
                        original_text = record.get('text', '')
                        if response_text:  # 只重试有内容的
                            task = self.analyze_llm_response(session, response_text, original_text, record_id)
                            retry_tasks.append(task)
                            retry_record_ids.append(record_id)
                
                if not retry_tasks:
                    logger.info("所有任务都已成功完成，无需重试")
                    break  # 没有需要重试的任务
                
                retry_count += 1
                logger.info(f"第 {retry_count} 次重试，重试 {len(retry_tasks)} 个失败任务")
                logger.info(f"失败任务记录ID: {retry_record_ids}")
                
                # 打印重试内容的详细信息
                for record_id in retry_record_ids:
                    record, _ = task_records[record_id]
                    original_text = record.get('text', '')[:100] + "..." if len(record.get('text', '')) > 100 else record.get('text', '')
                    response_text = record.get('response', '')[:200] + "..." if len(record.get('response', '')) > 200 else record.get('response', '')
                    logger.info(f"重试记录 {record_id}:")
                    logger.info(f"  原始内容: {original_text}")
                    logger.info(f"  LLM回答: {response_text}")
                
                # 执行重试，等待时间递增
                wait_time = min(5, retry_count)  # 最多等待5秒
                if retry_count > 1:
                    logger.info(f"等待 {wait_time} 秒后开始重试...")
                    await asyncio.sleep(wait_time)
                
                retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
                
                success_count = 0
                for i, result in enumerate(retry_results):
                    if isinstance(result, Exception):
                        logger.error(f"重试任务 {retry_record_ids[i]} 执行出错: {result}")
                        continue
                    
                    record_id, classification, error = result
                    if not error and classification:
                        # 重试成功
                        task_records[record_id] = (task_records[record_id][0], classification)
                        success_count += 1
                        logger.info(f"✅ 重试成功: 记录 {record_id} -> 分类 {classification}")
                    else:
                        logger.warning(f"❌ 重试失败: 记录 {record_id} -> 错误: {error}")
                        # 打印失败记录的内容，帮助诊断问题
                        record, _ = task_records[record_id]
                        original_text = record.get('text', '')[:150] + "..." if len(record.get('text', '')) > 150 else record.get('text', '')
                        logger.warning(f"   失败内容: {original_text}")
                
                logger.info(f"第 {retry_count} 次重试完成，成功 {success_count} 个，剩余失败 {len(retry_tasks) - success_count} 个")
        
        # 构建最终结果
        results = []
        failed_count = 0
        
        for i, (record, classification) in enumerate(task_records.values()):
            if classification is None:
                classification = '3'  # 默认值
                failed_count += 1
                # 记录最终失败的内容
                original_text = record.get('text', '')[:150] + "..." if len(record.get('text', '')) > 150 else record.get('text', '')
                logger.warning(f"⚠️ 最终失败记录 (使用默认值3): {original_text}")
            
            result_record = record.copy()
            result_record['llm_judgment_classification'] = classification
            results.append(result_record)
        
        logger.info(f"文件 {os.path.basename(file_path)} 处理完成")
        logger.info(f"总记录数: {len(results)}, 成功: {len(results) - failed_count}, 失败(使用默认值): {failed_count}")
        
        if failed_count > 0:
            logger.warning(f"注意：有 {failed_count} 条记录最终分析失败，已使用默认分类值 3")
        
        return results
    
    async def process_specified_files(self, json_dir: str, output_dir: str, file_list: list):
        """处理指定的JSON文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 处理指定文件列表
        logger.info(f"准备处理 {len(file_list)} 个指定的JSON文件")
        
        for json_file in file_list:
            try:
                file_path = os.path.join(json_dir, json_file)
                
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    logger.error(f"文件不存在: {file_path}")
                    continue
                    
                logger.info(f"正在处理: {json_file}")
                
                # 处理文件
                results = await self.process_single_json_file(file_path)
                
                # 保存分析结果
                output_file = os.path.join(output_dir, json_file.replace('.json', '_analyzed.json'))
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"已保存分析结果到: {output_file}")
                logger.info(f"完成文件 {json_file}，共处理 {len(results)} 条记录")
                
            except Exception as e:
                logger.error(f"处理文件 {json_file} 时出错: {e}")

async def main():
    # qwen API配置
    api_key = "put your api key here"
    base_url = "put your base url here"
    model = "qwen3-30b-a3b"
    
    # 输入和输出目录
    json_dir = "evaluation_results_batch"  # JSON文件所在目录
    output_dir = "analyzed_llm_results"  # 分析结果输出目录
    
    # 指定要分析的文件列表
    target_files = [
        "deepseek-r1-250528_detailed_public_health.json",
        "deepseek-r1-250528_detailed_respiratory.json", 
        "deepseek-r1-250528_no_guide.json",
        "deepseek-r1-250528_public_health_expert.json",
        "deepseek-r1-250528_respiratory_doctor.json",
        "qwen3-235b-a22b-thinking-2507_detailed_public_health.json",
        "qwen3-235b-a22b-thinking-2507_detailed_respiratory.json",
        "qwen3-235b-a22b-thinking-2507_no_guide.json", 
        "qwen3-235b-a22b-thinking-2507_public_health_expert.json",
        "qwen3-235b-a22b-thinking-2507_respiratory_doctor.json"
    ]
    
    logger.info("开始LLM回答分析...")
    logger.info(f"输入目录: {json_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"API模型: {model}")
    logger.info(f"并发数: 480")
    logger.info(f"待处理文件数: {len(target_files)}")
    
    # 创建分析器
    analyzer = LLMResponseAnalyzer(api_key, base_url, model)
    
    # 处理指定文件
    await analyzer.process_specified_files(json_dir, output_dir, target_files)
    
    logger.info("所有LLM文件分析完成！")
    
    # 统计结果
    if os.path.exists(output_dir):
        analyzed_files = [f for f in os.listdir(output_dir) if f.endswith('_analyzed.json')]
        logger.info(f"生成了 {len(analyzed_files)} 个分析结果文件:")
        for file in analyzed_files:
            logger.info(f"  - {file}")

if __name__ == "__main__":
    asyncio.run(main())