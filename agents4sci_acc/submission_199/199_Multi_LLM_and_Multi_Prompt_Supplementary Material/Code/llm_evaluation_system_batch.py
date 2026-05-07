import sqlite3
import json
import os
import time
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

from collections import defaultdict
import math

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 模型配置
MODELS_CONFIG = {
    # "mistral-large-latest": {
    #     "model": "put your model here",
    #     "key": "put your key here",
    #     "base_url": "put your base url here"
    # },
    # "chatgpt-4o-latest": {
    #     "model": "put your model here",
    #     "key": "put your key here",
    #     "base_url": "put your base url here"
    # },
    "qwen3-32b": {
        "model": "put your model here",
        "key": "put your key here",
        "base_url": "put your base url here"
    },
    # "llama-4-maverick": {
    #     "model": "put your model here",
    #     "key": "put your key here",
    #     "base_url": "put your base url here"
    # },
    # "deepseek-v3": {
    #     "model": "put your model here",
    #     "key": "put your key here",
    #     "base_url": "put your base url here"
    # },
    # "qwen3-235b": {
    #     "model": "put your model here",
    #     "key": "put your key here",
    #     "base_url": "put your base url here"
    # },
    # "doubao-seedream": {
    #     "model": "put your model here",
    #     "key": "put your key here",
    #     "base_url": "put your base url here"
    # },
    # "glm-4-airx": {
    #     "model": "put your model here",
    #     "key": "put your key here",
    #     "base_url": "put your base url here"
    # }
}

# 评估配置
EVALUATION_CONFIG = {
    "strategies": ["no_guide", "public_health_expert", "respiratory_doctor", "detailed_public_health", "detailed_respiratory"],
    "repetitions": 1,  # 只重复1次
    "max_workers": 320,  # 320并发：处理更多数据并发
    "request_timeout": 300,  # 5分钟超时，适应gemini较慢的响应速度
    "max_retry_attempts": 12,  # 最多重试12次
    "delay_between_requests": 0.05,  # 减少延迟
    "batch_size": 30,  # 每30条数据重试失败任务
    "total_records": 640  # 数据库总记录数
}

@dataclass
class EvaluationResult:
    """评估结果数据结构"""
    model_name: str
    strategy: str
    repetition: int
    weibo_id: str
    screen_name: str
    text: str
    created_at: str
    prompt: str
    response: str
    response_time: float
    timestamp: datetime
    temperature: float
    retry_count: int = 0
    token_usage: Optional[Dict] = None
    error: Optional[str] = None
    
    def to_dict(self):
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    


class PromptStrategy:
    """Prompt策略生成器"""
    
    @staticmethod
    def generate_prompt(strategy: str, text: str, screen_name: str = "", created_at: str = "") -> str:
        """根据策略生成相应的prompt"""
        
        if strategy == "no_guide":
            return f'请分析以下内容中是否存在错误："{text}"'
        
        elif strategy == "public_health_expert":
            return f'你是一名公共卫生专家，请分析以下内容中是否存在错误："{text}"'
        
        elif strategy == "respiratory_doctor":
            return f'你是一名呼吸科医生，请分析以下内容中是否存在错误："{text}"'
        
        elif strategy == "detailed_public_health":
            date_str = created_at.split()[0] if created_at else "未知日期"
            return f'你是一名公共卫生专家，这是新浪微博用户"{screen_name}"在{date_str}所发布的内容，请分析以下内容中是否存在错误："{text}"'
        
        elif strategy == "detailed_respiratory":
            date_str = created_at.split()[0] if created_at else "未知日期"
            return f'你是一名呼吸科医生，这是新浪微博用户"{screen_name}"在{date_str}所发布的内容，请分析以下内容中是否存在错误："{text}"'
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

class ModelHandler:
    """模型API调用处理器"""
    
    def __init__(self, model_name: str, config: Dict):
        self.model_name = model_name
        self.config = config
        
    async def call_model(self, prompt: str, session: aiohttp.ClientSession, repetition: int = 1) -> Tuple[str, float, Optional[Dict], Optional[str]]:
        """调用模型API"""
        start_time = time.time()
        
        # 固定temperature为0.5
        temperature = 0.5
        
        try:
            headers = {
                "Authorization": f"Bearer {self.config['key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 8000,
                "stream": False
            }
            
            async with session.post(
                f"{self.config['base_url']}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=EVALUATION_CONFIG["request_timeout"])
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    choice = result.get("choices", [{}])[0]
                    # 兼容不同返回格式
                    text_response = ""
                    if isinstance(choice, dict):
                        text_response = (
                            choice.get("message", {}).get("content")
                            or choice.get("text")
                            or choice.get("content")
                            or ""
                        )
                    elif isinstance(choice, str):
                        text_response = choice
                    
                    usage = result.get("usage", {})
                    return text_response, response_time, usage, None
                else:
                    error_text = await response.text()
                    return "", response_time, None, f"HTTP {response.status}: {error_text}"
                    
        except Exception as e:
            response_time = time.time() - start_time
            return "", response_time, None, str(e)

class BatchEvaluationSystem:
    """批处理LLM评估系统"""
    
    def __init__(self):
        self.results_dir = Path("evaluation_results_batch")
        self.results_dir.mkdir(exist_ok=True)
        

        
        self.reports_dir = self.results_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # 状态文件
        self.progress_file = self.results_dir / "progress.json"
        self.final_failed_file = self.results_dir / "final_failed_tasks.json"
        
        # 创建模型处理器
        self.model_handlers = {
            name: ModelHandler(name, config) 
            for name, config in MODELS_CONFIG.items()
        }
        

        
        # 加载进度
        self.progress = self.load_progress()
        

    
    def load_progress(self) -> dict:
        """加载进度"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    # 确保completed_tasks是set类型
                    if isinstance(progress_data.get("completed_tasks", []), list):
                        progress_data["completed_tasks"] = set(progress_data["completed_tasks"])
                    return progress_data
            except Exception as e:
                logger.warning(f"无法加载进度文件: {e}")
        
        return {
            "processed_records": 0,
            "current_batch": 0,
            "completed_tasks": set(),
            "failed_tasks_by_record": {},
            "final_failed_tasks": [],
            "processed_offset": 0  # 使用偏移量而不是记录ID
        }
    
    def save_progress(self):
        """保存进度"""
        try:
            progress_data = self.progress.copy()
            progress_data["completed_tasks"] = list(self.progress["completed_tasks"])
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"无法保存进度: {e}")
    
    def load_weibo_data_batch(self, start_offset: int = 0, batch_size: int = 10) -> List[Dict]:
        """批量加载微博数据（使用OFFSET而不是ID范围）"""
        try:
            conn = sqlite3.connect('llm.db')
            cursor = conn.cursor()
            
            # 使用OFFSET和LIMIT来分页查询
            query = """
            SELECT id, screen_name, text, created_at 
            FROM weibo 
            ORDER BY id
            LIMIT ? OFFSET ?
            """
            
            cursor.execute(query, (batch_size, start_offset))
            rows = cursor.fetchall()
            
            # 转换为字典列表
            weibo_data = []
            for row in rows:
                weibo_data.append({
                    "id": str(row[0]),
                    "screen_name": row[1],
                    "text": row[2],
                    "created_at": row[3]
                })
            
            conn.close()
            return weibo_data
            
        except Exception as e:
            logger.error(f"无法加载微博数据: {e}")
            return []
    
    def generate_task_id(self, model_name: str, strategy: str, repetition: int, weibo_id: str) -> str:
        """生成任务ID"""
        return f"{model_name}_{strategy}_{repetition}_{weibo_id}"
    
    def generate_tasks_for_record(self, weibo_data: Dict) -> List[Dict]:
        """为单条记录生成所有任务"""
        tasks = []
        for model_name in MODELS_CONFIG.keys():
            for strategy in EVALUATION_CONFIG["strategies"]:
                for repetition in range(1, EVALUATION_CONFIG["repetitions"] + 1):
                    task_id = self.generate_task_id(model_name, strategy, repetition, weibo_data["id"])
                    
                    # 检查是否已完成
                    if task_id in self.progress["completed_tasks"]:
                        continue
                    
                    tasks.append({
                        "task_id": task_id,
                        "model_name": model_name,
                        "strategy": strategy,
                        "repetition": repetition,
                        "weibo_data": weibo_data,
                        "retry_count": 0
                    })
        
        return tasks
    
    async def evaluate_single_task(self, task_data: Dict, session: aiohttp.ClientSession) -> EvaluationResult:
        """评估单个任务"""
        model_name = task_data["model_name"]
        strategy = task_data["strategy"]
        repetition = task_data["repetition"]
        weibo_data = task_data["weibo_data"]
        retry_count = task_data.get("retry_count", 0)
        
        # 生成prompt
        prompt = PromptStrategy.generate_prompt(
            strategy=strategy,
            text=weibo_data["text"],
            screen_name=weibo_data["screen_name"],
            created_at=weibo_data["created_at"]
        )
        
        # 调用模型
        model_handler = self.model_handlers[model_name]
        response, response_time, token_usage, error = await model_handler.call_model(prompt, session, repetition)
        
        # 固定temperature值
        temperature = 0.5
        
        # 创建结果
        result = EvaluationResult(
            model_name=model_name,
            strategy=strategy,
            repetition=repetition,
            weibo_id=weibo_data["id"],
            screen_name=weibo_data["screen_name"],
            text=weibo_data["text"],
            created_at=weibo_data["created_at"],
            prompt=prompt,
            response=response,
            response_time=response_time,
            timestamp=datetime.now(),
            temperature=temperature,
            retry_count=retry_count,
            token_usage=token_usage,
            error=error
        )
        
        return result
    
    def save_result_by_strategy(self, result: EvaluationResult):
        """保存结果到策略汇总JSON文件"""
        results_dir = Path("evaluation_results_batch")
        results_dir.mkdir(exist_ok=True)
        
        # 按模型和策略保存到单独的JSON文件
        filename = f"{result.model_name}_{result.strategy}.json"
        filepath = results_dir / filename
        
        try:
            # 读取现有数据
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []
            
            # 添加新结果
            data.append(result.to_dict())
            
            # 写回文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"无法保存结果到JSON {filepath}: {e}")

    async def retry_failed_tasks(self, failed_tasks: List[Dict], retry_attempt: int, session: aiohttp.ClientSession) -> List[Dict]:
        """重试失败的任务"""
        if not failed_tasks:
            return []
        
        logger.info(f"开始第 {retry_attempt} 次重试，共 {len(failed_tasks)} 个失败任务")
        
        # 准备重试任务
        retry_tasks = []
        for failed_task in failed_tasks:
            task_data = failed_task["task"].copy()
            task_data["retry_count"] = retry_attempt
            retry_tasks.append(task_data)
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(EVALUATION_CONFIG["max_workers"])
        
        async def retry_task(task_data):
            async with semaphore:
                try:
                    result = await self.evaluate_single_task(task_data, session)
                    
                    # 保存结果
                    self.save_result_by_strategy(result)
                    
                    # 记录完成状态
                    self.progress["completed_tasks"].add(task_data["task_id"])
                    
                    if result.error:
                        logger.warning(f"重试失败: {task_data['task_id']} - {result.error}")
                        return {"status": "failed", "task": task_data, "error": result.error}
                    else:
                        logger.info(f"重试成功: {task_data['task_id']} - {result.response_time:.2f}s")
                        return {"status": "success", "task": task_data}
                        
                except Exception as e:
                    logger.error(f"重试异常: {task_data['task_id']} - {e}")
                    return {"status": "failed", "task": task_data, "error": str(e)}
                finally:
                    await asyncio.sleep(EVALUATION_CONFIG["delay_between_requests"])
        
        # 执行重试任务
        results = await asyncio.gather(*[retry_task(task) for task in retry_tasks])
        
        # 统计结果
        successful = len([r for r in results if r["status"] == "success"])
        still_failed = [r for r in results if r["status"] == "failed"]
        
        logger.info(f"第 {retry_attempt} 次重试完成: 成功 {successful}, 仍失败 {len(still_failed)}")
        
        return still_failed
    
    async def process_records_batch(self, weibo_batch: List[Dict], session: aiohttp.ClientSession) -> List[Dict]:
        """处理一批记录（3条数据，120个并发任务）"""
        logger.info(f"开始处理 {len(weibo_batch)} 条记录")
        
        # 生成所有任务（3条记录 × 40任务/条 = 120个任务）
        all_tasks = []
        for weibo_data in weibo_batch:
            tasks = self.generate_tasks_for_record(weibo_data)
            all_tasks.extend(tasks)
        
        if not all_tasks:
            logger.info("所有任务都已完成")
            return []
        
        logger.info(f"生成了 {len(all_tasks)} 个任务，开始120并发执行")
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(EVALUATION_CONFIG["max_workers"])
        
        async def process_task(task_data):
            async with semaphore:
                try:
                    result = await self.evaluate_single_task(task_data, session)
                    
                    # 保存结果
                    self.save_result_by_strategy(result)
                    
                    # 记录完成状态
                    self.progress["completed_tasks"].add(task_data["task_id"])
                    
                    if result.error:
                        logger.warning(f"任务失败: {task_data['task_id']} - {result.error}")
                        return {"status": "failed", "task": task_data, "error": result.error}
                    else:
                        logger.info(f"任务完成: {task_data['task_id']} - {result.response_time:.2f}s")
                        return {"status": "success", "task": task_data}
                        
                except Exception as e:
                    logger.error(f"任务异常: {task_data['task_id']} - {e}")
                    return {"status": "failed", "task": task_data, "error": str(e)}
                finally:
                    await asyncio.sleep(EVALUATION_CONFIG["delay_between_requests"])
        
        # 执行所有任务
        results = await asyncio.gather(*[process_task(task) for task in all_tasks])
        
        # 收集失败任务
        failed_tasks = [r for r in results if r["status"] == "failed"]
        successful = len([r for r in results if r["status"] == "success"])
        
        logger.info(f"批次完成: 成功 {successful}, 失败 {len(failed_tasks)}")
        
        return failed_tasks

    async def run_batch_evaluation(self):
        """运行批处理评估"""
        logger.info("开始批处理LLM评估实验")
        logger.info(f"总数据量: {EVALUATION_CONFIG['total_records']} 条")
        logger.info(f"每条数据: {len(MODELS_CONFIG)} 模型 × {len(EVALUATION_CONFIG['strategies'])} 策略 × {EVALUATION_CONFIG['repetitions']} 重复 = {len(MODELS_CONFIG) * len(EVALUATION_CONFIG['strategies']) * EVALUATION_CONFIG['repetitions']} 个任务")
        total_tasks = EVALUATION_CONFIG['total_records'] * len(MODELS_CONFIG) * len(EVALUATION_CONFIG['strategies']) * EVALUATION_CONFIG['repetitions']
        logger.info(f"总任务数: {total_tasks}")
        logger.info(f"240并发策略：每轮240个任务，每3轮重试失败任务")
        
        # 生成所有任务
        all_tasks = []
        start_offset = self.progress.get("processed_offset", 0)
        
        # 加载所有微博数据
        for offset in range(start_offset, EVALUATION_CONFIG["total_records"], 50):  # 每次加载50条
            batch_size = min(50, EVALUATION_CONFIG["total_records"] - offset)
            weibo_batch = self.load_weibo_data_batch(offset, batch_size)
            
            for weibo_data in weibo_batch:
                tasks = self.generate_tasks_for_record(weibo_data)
                all_tasks.extend(tasks)
        
        if not all_tasks:
            logger.info("所有任务都已完成")
            self.generate_completion_report()
            return
        
        logger.info(f"生成了 {len(all_tasks)} 个待处理任务")
        
        # 按240个任务分组
        task_batch_size = 320
        task_batches = [all_tasks[i:i + task_batch_size] for i in range(0, len(all_tasks), task_batch_size)]
        logger.info(f"分为 {len(task_batches)} 轮，每轮最多 {task_batch_size} 个任务")
        
        # 收集失败任务
        all_failed_tasks = []
        round_counter = 0
        
        async with aiohttp.ClientSession() as session:
            # 处理每一轮任务
            for batch_idx, task_batch in enumerate(task_batches):
                round_counter += 1
                logger.info(f"开始第 {round_counter} 轮: {len(task_batch)} 个任务")
                
                # 240并发处理当前批次
                failed_tasks = await self.process_task_batch_concurrent(task_batch, session, round_counter)
                
                if failed_tasks:
                    all_failed_tasks.extend(failed_tasks)
                    logger.info(f"第 {round_counter} 轮完成，失败 {len(failed_tasks)} 个任务")
                else:
                    logger.info(f"第 {round_counter} 轮完成，全部成功")
                
                # 每3轮重试失败任务
                if round_counter % 3 == 0 and all_failed_tasks:
                    logger.info(f"已完成 {round_counter} 轮，开始重试累积的 {len(all_failed_tasks)} 个失败任务")
                    
                    current_failed = all_failed_tasks
                    for retry_attempt in range(1, EVALUATION_CONFIG["max_retry_attempts"] + 1):
                        current_failed = await self.retry_failed_tasks(current_failed, retry_attempt, session)
                        
                        if not current_failed:
                            logger.info(f"所有失败任务已在第 {retry_attempt} 次重试中成功")
                            break
                    
                    # 更新失败任务列表
                    all_failed_tasks = current_failed
                    if current_failed:
                        logger.warning(f"重试后仍有 {len(current_failed)} 个任务失败")
                
                # 保存进度
                self.progress["current_batch"] = round_counter
                self.progress["processed_records"] = min(EVALUATION_CONFIG["total_records"], 
                                                       (batch_idx + 1) * task_batch_size // (len(MODELS_CONFIG) * len(EVALUATION_CONFIG["strategies"])))
                self.save_progress()
        
        # 最终重试剩余失败任务
        if all_failed_tasks:
            logger.info(f"所有轮次完成，最终重试剩余的 {len(all_failed_tasks)} 个失败任务")
            
            async with aiohttp.ClientSession() as session:
                current_failed = all_failed_tasks
                for retry_attempt in range(1, EVALUATION_CONFIG["max_retry_attempts"] + 1):
                    current_failed = await self.retry_failed_tasks(current_failed, retry_attempt, session)
                    
                    if not current_failed:
                        logger.info(f"最终重试：所有失败任务已在第 {retry_attempt} 次重试中成功")
                        break
                
                if current_failed:
                    self.generate_final_failure_report(current_failed)
        
        # 生成完成报告
        self.generate_completion_report()
        logger.info("批处理评估实验完成")
    
    async def process_task_batch_concurrent(self, task_batch: List[Dict], session: aiohttp.ClientSession, round_num: int) -> List[Dict]:
        """240并发处理一批任务"""
        logger.info(f"第 {round_num} 轮开始，240并发处理 {len(task_batch)} 个任务")
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(EVALUATION_CONFIG["max_workers"])
        
        async def process_task(task_data):
            async with semaphore:
                try:
                    result = await self.evaluate_single_task(task_data, session)
                    
                    # 保存结果
                    self.save_result_by_strategy(result)
                    
                    # 记录完成状态
                    self.progress["completed_tasks"].add(task_data["task_id"])
                    
                    if result.error:
                        logger.warning(f"任务失败: {task_data['task_id']} - {result.error}")
                        return {"status": "failed", "task": task_data, "error": result.error}
                    else:
                        logger.info(f"任务完成: {task_data['task_id']} - {result.response_time:.2f}s")
                        return {"status": "success", "task": task_data}
                        
                except Exception as e:
                    logger.error(f"任务异常: {task_data['task_id']} - {e}")
                    return {"status": "failed", "task": task_data, "error": str(e)}
                finally:
                    await asyncio.sleep(EVALUATION_CONFIG["delay_between_requests"])
        
        # 240并发执行所有任务
        results = await asyncio.gather(*[process_task(task) for task in task_batch])
        
        # 收集失败任务
        failed_tasks = [r for r in results if r["status"] == "failed"]
        successful = len([r for r in results if r["status"] == "success"])
        
        logger.info(f"第 {round_num} 轮完成: 成功 {successful}, 失败 {len(failed_tasks)}")
        
        return failed_tasks
    
    def generate_final_failure_report(self, failed_tasks: List[Dict]):
        """生成最终失败报告"""
        logger.info(f"生成最终失败报告，共 {len(failed_tasks)} 个失败任务")
        
        # 按模型统计失败情况
        failure_stats = {
            "总失败任务数": len(failed_tasks),
            "生成时间": datetime.now().isoformat(),
            "失败任务详情": [],
            "按模型统计": defaultdict(int),
            "按策略统计": defaultdict(int),
            "按错误类型统计": defaultdict(int)
        }
        
        for failed_task in failed_tasks:
            task = failed_task["task"]
            error = failed_task["error"]
            
            failure_stats["失败任务详情"].append({
                "task_id": task["task_id"],
                "model_name": task["model_name"],
                "strategy": task["strategy"],
                "repetition": task["repetition"],
                "weibo_id": task["weibo_data"]["id"],
                "error": error,
                "retry_count": task.get("retry_count", 0)
            })
            
            failure_stats["按模型统计"][task["model_name"]] += 1
            failure_stats["按策略统计"][task["strategy"]] += 1
            
            # 分类错误类型
            if "HTTP" in error:
                failure_stats["按错误类型统计"]["HTTP错误"] += 1
            elif "timeout" in error.lower():
                failure_stats["按错误类型统计"]["超时错误"] += 1
            elif "connect" in error.lower():
                failure_stats["按错误类型统计"]["连接错误"] += 1
            else:
                failure_stats["按错误类型统计"]["其他错误"] += 1
        
        # 转换defaultdict为普通dict
        failure_stats["按模型统计"] = dict(failure_stats["按模型统计"])
        failure_stats["按策略统计"] = dict(failure_stats["按策略统计"])
        failure_stats["按错误类型统计"] = dict(failure_stats["按错误类型统计"])
        
        # 保存失败报告
        try:
            with open(self.final_failed_file, 'w', encoding='utf-8') as f:
                json.dump(failure_stats, f, ensure_ascii=False, indent=2)
                
            # 生成可读的失败报告
            report_file = self.reports_dir / f"failure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# 最终失败任务报告\n\n")
                f.write(f"**生成时间**: {failure_stats['生成时间']}\n")
                f.write(f"**总失败任务数**: {failure_stats['总失败任务数']}\n\n")
                
                f.write("## 按模型统计\n")
                for model, count in failure_stats["按模型统计"].items():
                    f.write(f"- {model}: {count} 个失败任务\n")
                
                f.write("\n## 按策略统计\n")
                for strategy, count in failure_stats["按策略统计"].items():
                    f.write(f"- {strategy}: {count} 个失败任务\n")
                
                f.write("\n## 按错误类型统计\n")
                for error_type, count in failure_stats["按错误类型统计"].items():
                    f.write(f"- {error_type}: {count} 个失败任务\n")
                
                f.write("\n## 详细失败任务\n")
                for task in failure_stats["失败任务详情"]:
                    f.write(f"- **{task['task_id']}**: {task['error']}\n")
            
            logger.info(f"失败报告已保存到: {report_file}")
            
        except Exception as e:
            logger.error(f"无法保存失败报告: {e}")
    
    def generate_completion_report(self):
        """生成完成报告"""
        logger.info("生成完成报告")
        
        # 统计策略JSON文件
        total_records = 0
        strategy_stats = {}
        results_dir = Path("evaluation_results_batch")
        
        for model_name in MODELS_CONFIG.keys():
            for strategy in EVALUATION_CONFIG["strategies"]:
                strategy_file = results_dir / f"{model_name}_{strategy}.json"
                if strategy_file.exists():
                    try:
                        with open(strategy_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            count = len(data)
                            if strategy not in strategy_stats:
                                strategy_stats[strategy] = 0
                            strategy_stats[strategy] += count
                            total_records += count
                    except Exception as e:
                        logger.warning(f"无法读取策略文件 {strategy_file}: {e}")
                        if strategy not in strategy_stats:
                            strategy_stats[strategy] = 0
                else:
                    if strategy not in strategy_stats:
                        strategy_stats[strategy] = 0
        
        completion_stats = {
            "完成时间": datetime.now().isoformat(),
            "处理记录数": self.progress["processed_records"],
            "完成任务数": len(self.progress["completed_tasks"]),
            "总记录数": total_records,
            "按策略统计": strategy_stats,
            "预期总任务数": EVALUATION_CONFIG["total_records"] * len(MODELS_CONFIG) * len(EVALUATION_CONFIG["strategies"]) * EVALUATION_CONFIG["repetitions"],
            "完成率": f"{(total_records / (EVALUATION_CONFIG['total_records'] * len(MODELS_CONFIG) * len(EVALUATION_CONFIG['strategies']) * EVALUATION_CONFIG['repetitions']) * 100):.2f}%"
        }
        
        # 保存完成报告
        try:
            completion_file = self.reports_dir / f"completion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(completion_file, 'w', encoding='utf-8') as f:
                json.dump(completion_stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"完成报告已保存到: {completion_file}")
            
        except Exception as e:
            logger.error(f"无法保存完成报告: {e}")
        
        return completion_stats

async def main():
    """主函数"""
    system = BatchEvaluationSystem()
    
    # 运行批处理评估
    await system.run_batch_evaluation()
    
    print("\n" + "="*60)
    print("批处理LLM评估实验完成")
    print("="*60)
    print(f"处理记录数: {system.progress['processed_records']}")
    print(f"完成任务数: {len(system.progress['completed_tasks'])}")
    print(f"JSON结果位置: evaluation_results_batch/")
    print(f"报告文件位置: {system.reports_dir}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main()) 