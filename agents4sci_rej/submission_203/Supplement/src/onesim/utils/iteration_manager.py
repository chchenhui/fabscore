# -*- coding: utf-8 -*-
"""
迭代管理器模块
管理调试修复的迭代过程，跟踪修复进度和历史
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import json
from pathlib import Path


class IterationStatus(Enum):
    """迭代状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MAX_ATTEMPTS_REACHED = "max_attempts_reached"


@dataclass
class IterationResult:
    """单次迭代结果"""
    iteration_id: int
    status: IterationStatus
    start_time: float
    end_time: Optional[float] = None
    errors_found: List[Dict[str, Any]] = field(default_factory=list)
    fixes_applied: List[Dict[str, Any]] = field(default_factory=list)
    docker_output: Dict[str, str] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """获取迭代持续时间"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class DebuggingSession:
    """调试会话"""
    session_id: str
    env_name: str
    start_time: float
    max_iterations: int = 5
    timeout_per_iteration: int = 300
    iterations: List[IterationResult] = field(default_factory=list)
    final_status: Optional[IterationStatus] = None
    total_fixes_applied: int = 0
    
    @property
    def current_iteration(self) -> int:
        """当前迭代次数"""
        return len(self.iterations)
    
    @property
    def is_complete(self) -> bool:
        """是否已完成"""
        return self.final_status is not None
    
    @property
    def is_successful(self) -> bool:
        """是否成功"""
        return self.final_status == IterationStatus.SUCCESS


class IterationManager:
    """迭代管理器"""
    
    def __init__(self,
                 max_iterations: int = 5,
                 timeout_per_iteration: int = 300,
                 save_history: bool = True,
                 history_dir: Optional[str] = None):
        """
        初始化迭代管理器
        
        Args:
            max_iterations: 最大迭代次数
            timeout_per_iteration: 每次迭代的超时时间
            save_history: 是否保存历史记录
            history_dir: 历史记录保存目录
        """
        self.max_iterations = max_iterations
        self.timeout_per_iteration = timeout_per_iteration
        self.save_history = save_history
        self.history_dir = Path(history_dir) if history_dir else Path("./debug_history")
        
        # 确保历史目录存在
        if self.save_history:
            self.history_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session: Optional[DebuggingSession] = None
        
    def start_session(self, session_id: str, env_name: str) -> DebuggingSession:
        """
        开始新的调试会话
        
        Args:
            session_id: 会话ID
            env_name: 环境名称
            
        Returns:
            DebuggingSession: 调试会话对象
        """
        self.current_session = DebuggingSession(
            session_id=session_id,
            env_name=env_name,
            start_time=time.time(),
            max_iterations=self.max_iterations,
            timeout_per_iteration=self.timeout_per_iteration
        )
        
        logger.info(f"开始调试会话: {session_id}, 环境: {env_name}")
        return self.current_session
    
    def start_iteration(self) -> IterationResult:
        """
        开始新的迭代
        
        Returns:
            IterationResult: 迭代结果对象
        """
        if not self.current_session:
            raise RuntimeError("没有活跃的调试会话")
        
        if self.current_session.current_iteration >= self.max_iterations:
            raise RuntimeError("已达到最大迭代次数")
        
        iteration_id = self.current_session.current_iteration + 1
        iteration = IterationResult(
            iteration_id=iteration_id,
            status=IterationStatus.RUNNING,
            start_time=time.time()
        )
        
        self.current_session.iterations.append(iteration)
        logger.info(f"开始第 {iteration_id} 次迭代")
        
        return iteration
    
    def complete_iteration(self,
                         iteration: IterationResult,
                         status: IterationStatus,
                         errors_found: List[Dict[str, Any]] = None,
                         fixes_applied: List[Dict[str, Any]] = None,
                         docker_output: Dict[str, str] = None,
                         error_message: Optional[str] = None) -> None:
        """
        完成当前迭代
        
        Args:
            iteration: 迭代对象
            status: 最终状态
            errors_found: 发现的错误列表
            fixes_applied: 应用的修复列表
            docker_output: Docker输出
            error_message: 错误消息
        """
        iteration.end_time = time.time()
        iteration.status = status
        iteration.success = (status == IterationStatus.SUCCESS)
        iteration.error_message = error_message
        
        if errors_found:
            iteration.errors_found = errors_found
        if fixes_applied:
            iteration.fixes_applied = fixes_applied
            self.current_session.total_fixes_applied += len(fixes_applied)
        if docker_output:
            iteration.docker_output = docker_output
        
        iteration.execution_time = iteration.duration
        
        logger.info(f"第 {iteration.iteration_id} 次迭代完成, 状态: {status.value}")
        
        # 保存历史记录
        if self.save_history:
            self._save_iteration_history(iteration)
    
    def should_continue(self) -> Tuple[bool, str]:
        """
        判断是否应该继续迭代
        
        Returns:
            Tuple[bool, str]: (是否继续, 原因)
        """
        if not self.current_session:
            return False, "没有活跃的会话"
        
        # 检查是否已成功
        last_iteration = self.current_session.iterations[-1] if self.current_session.iterations else None
        if last_iteration and last_iteration.status == IterationStatus.SUCCESS:
            return False, "Debugging completed successfully"
        
        # 检查是否达到最大迭代次数
        if self.current_session.current_iteration >= self.max_iterations:
            return False, f"Maximum number of iterations reached ({self.max_iterations})"
        
        # 检查连续失败次数
        consecutive_failures = self._count_consecutive_failures()
        if consecutive_failures >= 3:
            return False, f"Failed {consecutive_failures} times in a row"
        
        return True, "Continue iteration"
    
    def finish_session(self, final_status: IterationStatus) -> DebuggingSession:
        """
        结束调试会话
        
        Args:
            final_status: 最终状态
            
        Returns:
            DebuggingSession: 完成的会话对象
        """
        if not self.current_session:
            raise RuntimeError("没有活跃的调试会话")
        
        self.current_session.final_status = final_status
        
        logger.info(f"调试会话结束: {self.current_session.session_id}, "
                   f"最终状态: {final_status.value}, "
                   f"总迭代次数: {self.current_session.current_iteration}, "
                   f"总修复数: {self.current_session.total_fixes_applied}")
        
        # 保存完整会话历史
        if self.save_history:
            self._save_session_history()
        
        completed_session = self.current_session
        self.current_session = None
        
        return completed_session
    
    def get_session_summary(self) -> Dict[str, Any]:
        """获取会话摘要"""
        if not self.current_session:
            return {"error": "没有活跃的会话"}
        
        return {
            "session_id": self.current_session.session_id,
            "env_name": self.current_session.env_name,
            "current_iteration": self.current_session.current_iteration,
            "max_iterations": self.current_session.max_iterations,
            "total_fixes_applied": self.current_session.total_fixes_applied,
            "is_complete": self.current_session.is_complete,
            "final_status": self.current_session.final_status.value if self.current_session.final_status else None,
            "iterations": [
                {
                    "iteration_id": it.iteration_id,
                    "status": it.status.value,
                    "duration": it.duration,
                    "errors_count": len(it.errors_found),
                    "fixes_count": len(it.fixes_applied),
                    "success": it.success
                }
                for it in self.current_session.iterations
            ]
        }
    
    def _count_consecutive_failures(self) -> int:
        """计算连续失败次数"""
        if not self.current_session or not self.current_session.iterations:
            return 0
        
        consecutive_failures = 0
        for iteration in reversed(self.current_session.iterations):
            if iteration.status in [IterationStatus.FAILED, IterationStatus.TIMEOUT]:
                consecutive_failures += 1
            else:
                break
        
        return consecutive_failures
    
    def _save_iteration_history(self, iteration: IterationResult) -> None:
        """保存迭代历史"""
        try:
            if not self.current_session:
                return
                
            session_dir = self.history_dir / self.current_session.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            iteration_file = session_dir / f"iteration_{iteration.iteration_id}.json"
            
            iteration_data = {
                "iteration_id": iteration.iteration_id,
                "status": iteration.status.value,
                "start_time": iteration.start_time,
                "end_time": iteration.end_time,
                "duration": iteration.duration,
                "success": iteration.success,
                "error_message": iteration.error_message,
                "errors_found": iteration.errors_found,
                "fixes_applied": iteration.fixes_applied,
                "docker_output": iteration.docker_output
            }
            
            with open(iteration_file, 'w', encoding='utf-8') as f:
                json.dump(iteration_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存迭代历史失败: {e}")
    
    def _save_session_history(self) -> None:
        """保存会话历史"""
        try:
            if not self.current_session:
                return
                
            session_dir = self.history_dir / self.current_session.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            session_file = session_dir / "session_summary.json"
            
            session_data = {
                "session_id": self.current_session.session_id,
                "env_name": self.current_session.env_name,
                "start_time": self.current_session.start_time,
                "end_time": time.time(),
                "total_duration": time.time() - self.current_session.start_time,
                "max_iterations": self.current_session.max_iterations,
                "actual_iterations": self.current_session.current_iteration,
                "total_fixes_applied": self.current_session.total_fixes_applied,
                "final_status": self.current_session.final_status.value,
                "is_successful": self.current_session.is_successful,
                "iterations_summary": [
                    {
                        "iteration_id": it.iteration_id,
                        "status": it.status.value,
                        "duration": it.duration,
                        "errors_count": len(it.errors_found),
                        "fixes_count": len(it.fixes_applied),
                        "success": it.success
                    }
                    for it in self.current_session.iterations
                ]
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存会话历史失败: {e}")
    
    def load_session_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        加载会话历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            Optional[Dict[str, Any]]: 会话历史数据
        """
        try:
            session_file = self.history_dir / session_id / "session_summary.json"
            if not session_file.exists():
                return None
                
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"加载会话历史失败: {e}")
            return None
    
    def list_available_sessions(self) -> List[str]:
        """获取可用的会话列表"""
        try:
            if not self.history_dir.exists():
                return []
                
            sessions = []
            for session_dir in self.history_dir.iterdir():
                if session_dir.is_dir() and (session_dir / "session_summary.json").exists():
                    sessions.append(session_dir.name)
                    
            return sorted(sessions)
            
        except Exception as e:
            logger.error(f"获取会话列表失败: {e}")
            return []


# 便捷函数
def create_session_manager(max_iterations: int = 5,
                         timeout_per_iteration: int = 300) -> IterationManager:
    """
    创建迭代管理器的便捷函数
    
    Args:
        max_iterations: 最大迭代次数
        timeout_per_iteration: 每次迭代超时时间
        
    Returns:
        IterationManager: 迭代管理器实例
    """
    return IterationManager(
        max_iterations=max_iterations,
        timeout_per_iteration=timeout_per_iteration
    )


async def run_with_timeout(coro, timeout: int):
    """
    带超时的协程执行
    
    Args:
        coro: 协程对象
        timeout: 超时时间
        
    Returns:
        协程结果或抛出超时异常
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"操作超时 ({timeout}秒)")
        raise