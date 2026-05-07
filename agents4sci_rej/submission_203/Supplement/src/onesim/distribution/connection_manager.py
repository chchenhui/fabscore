import asyncio
import grpc
import time
import json
import weakref
import random
from typing import Dict, Any, Optional, Tuple, List, Set
from loguru import logger

class CircuitBreaker:
    """熔断器实现，用于防止对故障服务的持续请求"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=30.0, half_open_success=2):
        self.failure_threshold = failure_threshold  # 触发熔断的失败次数
        self.recovery_timeout = recovery_timeout    # 熔断后尝试恢复的时间(秒)
        self.half_open_success = half_open_success  # 半开状态需要成功次数
        
        self.failures = 0                # 当前失败计数
        self.state = "closed"            # 熔断器状态: closed(正常), open(熔断), half-open(尝试恢复)
        self.last_failure_time = 0       # 最后一次失败时间
        self.successes_since_half_open = 0  # 半开状态下的成功次数
    
    def record_failure(self):
        """记录一次失败"""
        self.failures += 1
        self.last_failure_time = time.time()
        
        # 检查是否需要触发熔断
        if self.state == "closed" and self.failures >= self.failure_threshold:
            logger.warning(f"Circuit breaker tripped after {self.failures} failures")
            self.state = "open"
        
        # 如果熔断器已经半开且失败，重新熔断
        elif self.state == "half-open":
            logger.warning("Circuit breaker reopened after failure in half-open state")
            self.state = "open"
            self.successes_since_half_open = 0
    
    def record_success(self):
        """记录一次成功"""
        # 只有在半开状态时才需要记录成功
        if self.state == "half-open":
            self.successes_since_half_open += 1
            
            # 检查是否达到恢复阈值
            if self.successes_since_half_open >= self.half_open_success:
                logger.info(f"Circuit breaker reset after {self.successes_since_half_open} consecutive successes")
                self.reset()
    
    def reset(self):
        """重置熔断器状态"""
        self.failures = 0
        self.state = "closed"
        self.last_failure_time = 0
        self.successes_since_half_open = 0
    
    def is_allowed(self):
        """检查当前是否允许请求通过"""
        # 如果熔断器关闭(正常状态)，允许请求
        if self.state == "closed":
            return True
        
        # 如果熔断器打开，检查是否达到恢复时间
        if self.state == "open":
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                # 进入半开状态，允许少量请求尝试
                logger.info(f"Circuit half-open after {elapsed:.1f}s timeout")
                self.state = "half-open"
                self.successes_since_half_open = 0
                return True
            return False
        
        # 半开状态限制并发请求数，随机允许约20%的请求
        if self.state == "half-open":
            return random.random() < 0.2
        
        return True  # 默认安全情况


class ConnectionShard:
    """
    连接管理的分片，处理一部分连接的创建和维护
    """
    
    def __init__(self, 
                shard_id: int, 
                semaphore_size: int = 100, 
                lock_timeout: float = 5.0,
                max_idle_time: float = 3600.0):
        """
        初始化连接分片
        
        Args:
            shard_id: 分片标识符
            semaphore_size: 信号量大小，控制并发访问数
            lock_timeout: 获取锁的超时时间(秒)
            max_idle_time: 闲置连接的最大保留时间(秒)
        """
        self.shard_id = shard_id
        
        # 连接池相关
        self.channels = {}  # {(address, port): (channel, last_used_time, use_count)}
        self.stubs = {}     # {(address, port, stub_type): stub}
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(semaphore_size)
        self.lock_timeout = lock_timeout
        
        # 连接管理
        self.max_idle_time = max_idle_time
        self.cleanup_task = None
        
        # 熔断器 {(address, port): CircuitBreaker}
        self.circuit_breakers = {}
        
        # 已知可用的连接集合，避免重复检查
        self.known_good_connections = set()  # {(address, port)}
        
        # 预连接标记
        self.preconnected = False
    
    async def get_channel(self, address: str, port: int) -> grpc.aio.Channel:
        """
        获取或创建指定地址和端口的通道
        
        Args:
            address: 目标地址
            port: 目标端口
            
        Returns:
            gRPC 异步通道
        """
        key = (address, port)
        
        # 熔断器检查 - 如果连接被熔断，直接拒绝
        if key in self.circuit_breakers and not self.circuit_breakers[key].is_allowed():
            raise RuntimeError(f"Circuit breaker open for {key}, service may be unavailable")
        
        # 使用信号量控制并发
        acquired = False
        try:
            # 尝试获取信号量，设置超时
            try:
                acquired = await asyncio.wait_for(self.semaphore.acquire(), timeout=self.lock_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Shard {self.shard_id}: Timeout acquiring semaphore for {key}, creating uncached channel")
                # 超时情况下创建临时通道并返回，但不缓存
                return self._create_channel(address, port, cache=False)
                
            if acquired:
                try:
                    # 先检查是否有缓存的通道且是已知良好的连接
                    if key in self.channels and key in self.known_good_connections:
                        channel, last_used, use_count = self.channels[key]
                        
                        # 更新使用统计并返回
                        self.channels[key] = (channel, time.time(), use_count + 1)
                        return channel
                    
                    # 检查是否已有缓存的通道但需要验证状态
                    if key in self.channels:
                        channel, last_used, use_count = self.channels[key]
                        
                        # 检查通道状态，确保它是可用的
                        current_state = channel.get_state(try_to_connect=False)
                        
                        # 如果通道已准备好，直接使用
                        if current_state == grpc.ChannelConnectivity.READY:
                            self.channels[key] = (channel, time.time(), use_count + 1)
                            self.known_good_connections.add(key)
                            return channel
                            
                        # 如果通道处于IDLE状态且最近使用过，尝试激活它
                        elif current_state == grpc.ChannelConnectivity.IDLE and time.time() - last_used < 300:
                            # 尝试等待通道就绪 - 但使用更短的超时
                            try:
                                is_ready = await self._wait_for_ready_state(channel, timeout=10.0)
                                if is_ready:
                                    # 通道现在已准备好
                                    self.channels[key] = (channel, time.time(), use_count + 1)
                                    self.known_good_connections.add(key)
                                    return channel
                            except Exception as e:
                                logger.debug(f"Shard {self.shard_id}: Error checking idle channel: {e}")
                        
                        # 其他状态或激活失败，创建新通道
                        logger.debug(f"Shard {self.shard_id}: Channel for {key} in state {current_state}, creating new one")
                        
                        # 移除旧通道但不等待关闭完成，避免阻塞
                        old_channel = self.channels.pop(key)[0]
                        asyncio.create_task(old_channel.close())
                    
                    # 创建新通道并缓存
                    channel = self._create_channel(address, port)
                    
                    # 尝试快速检查通道状态 - 设置较短超时，避免阻塞太久
                    is_ready = False
                    try:
                        is_ready = await self._wait_for_ready_state(channel, timeout=10.0)
                    except Exception as e:
                        logger.debug(f"Shard {self.shard_id}: Error checking new channel: {e}")
                    
                    # 无论是否就绪都缓存，下次使用时会再次检查
                    self.channels[key] = (channel, time.time(), 1)
                    
                    if is_ready:
                        self.known_good_connections.add(key)
                        
                        # 设置弱引用关闭回调
                        weak_self = weakref.ref(self)
                        if hasattr(channel, "add_on_close_callback"):
                            channel.add_on_close_callback(
                                lambda: asyncio.create_task(
                                    weak_self()._on_channel_close(key) if weak_self() else None
                                )
                            )
                    
                    # 如果通道创建成功，重置熔断器（如果存在）
                    if key in self.circuit_breakers:
                        self.circuit_breakers[key].record_success()
                        
                    return channel
                    
                finally:
                    # 确保释放信号量
                    self.semaphore.release()
                    
        except Exception as e:
            logger.error(f"Shard {self.shard_id}: Unexpected error in get_channel for {key}: {e}")
            
            # 记录失败到熔断器
            if key in self.circuit_breakers:
                self.circuit_breakers[key].record_failure()
            else:
                # 创建新熔断器
                self.circuit_breakers[key] = CircuitBreaker()
                self.circuit_breakers[key].record_failure()
                
            # 确保释放信号量
            if acquired:
                self.semaphore.release()
                
            # 出错情况下创建临时通道
            return self._create_channel(address, port, cache=False)
    
    def _create_channel(self, address: str, port: int, cache: bool = True) -> grpc.aio.Channel:
        """
        创建gRPC通道，配置优化的参数
        
        Args:
            address: 目标地址
            port: 目标端口
            cache: 是否需要缓存此通道
            
        Returns:
            gRPC异步通道
        """
        # 创建新通道配置 - 优化以提高稳定性
        options = [
            ('grpc.keepalive_time_ms', 10000),           # 每10秒发送一次keepalive
            ('grpc.keepalive_timeout_ms', 5000),         # 5秒等待keepalive响应
            ('grpc.http2.max_pings_without_data', 5),    # 允许无数据ping
            ('grpc.http2.max_ping_strikes', 0),          # 不因ping断开连接
            ('grpc.http2.min_ping_interval_without_data_ms', 30000),
            ('grpc.keepalive_permit_without_calls', 1),  # 空闲时允许keepalive
            ('grpc.max_connection_idle_ms', 600000),     # 10分钟空闲超时
            ('grpc.max_connection_age_ms', 1800000),     # 30分钟最大年龄
            ('grpc.max_connection_age_grace_ms', 60000), # 1分钟关闭宽限期
            ('grpc.enable_retries', 1),                  # 启用重试
            ('grpc.min_reconnect_backoff_ms', 100),      # 最小重连间隔100ms
            ('grpc.max_reconnect_backoff_ms', 5000),     # 最大重连间隔5秒
            ('grpc.initial_reconnect_backoff_ms', 100),  # 初始重连间隔
            ('grpc.client_idle_timeout_ms', -1),         # 禁用客户端空闲超时
            ('grpc.service_config', json.dumps({
                'methodConfig': [{
                    'name': [{'service': 'agent.AgentService'}],
                    'retryPolicy': {
                        'maxAttempts': 5,
                        'initialBackoff': '0.1s',
                        'maxBackoff': '30s',
                        'backoffMultiplier': 2.0,
                        'retryableStatusCodes': [
                            'UNAVAILABLE', 'DEADLINE_EXCEEDED', 
                            'RESOURCE_EXHAUSTED', 'CANCELLED'
                        ]
                    }
                }]
            }))
        ]
        
        return grpc.aio.insecure_channel(f"{address}:{port}", options=options)
                
    async def _wait_for_ready_state(self, channel, timeout=10.0):
        """
        等待通道达到READY状态 - 优化版本，使用更短的超时和间隔
        
        Args:
            channel: 需要等待的gRPC通道
            timeout: 最大等待时间（秒）
            
        Returns:
            bool: 如果通道变为READY则返回True，否则返回False
        """
        start_time = time.time()
        end_time = start_time + timeout
        
        # 首先检查当前状态
        current_state = channel.get_state(try_to_connect=True)
        if current_state == grpc.ChannelConnectivity.READY:
            return True
        
        # 循环等待，直到达到READY状态或超时
        while time.time() < end_time:
            try:
                # 计算剩余时间
                remaining = end_time - time.time()
                if remaining <= 0:
                    break
                    
                # 等待状态变化，每次最多等待0.5秒 - 避免长时间阻塞
                await asyncio.wait_for(
                    channel.wait_for_state_change(current_state),
                    timeout=min(0.5, remaining)
                )
                
                # 检查新状态
                new_state = channel.get_state(try_to_connect=False)
                
                if new_state == grpc.ChannelConnectivity.READY:
                    return True
                elif new_state in (grpc.ChannelConnectivity.TRANSIENT_FAILURE, grpc.ChannelConnectivity.SHUTDOWN):
                    return False
                    
                # 更新当前状态并继续等待
                current_state = new_state
                
            except asyncio.TimeoutError:
                # 短暂超时，但总等待时间内，继续检查
                current_state = channel.get_state(try_to_connect=True)
                
                if current_state == grpc.ChannelConnectivity.READY:
                    return True
                elif current_state in (grpc.ChannelConnectivity.TRANSIENT_FAILURE, grpc.ChannelConnectivity.SHUTDOWN):
                    return False
        
        # 检查最终状态
        final_state = channel.get_state(try_to_connect=False)
        return final_state == grpc.ChannelConnectivity.READY
    
    async def _on_channel_close(self, key):
        """处理通道关闭回调"""
        # 使用异步任务避免阻塞
        try:
            # 如果存在于已知良好连接集合，移除
            if key in self.known_good_connections:
                self.known_good_connections.remove(key)
                
            async with self.semaphore:
                if key in self.channels:
                    self.channels.pop(key, None)
                    
                    # 移除相关存根
                    stub_keys = [k for k in self.stubs.keys() if k[0] == key[0] and k[1] == key[1]]
                    for stub_key in stub_keys:
                        self.stubs.pop(stub_key, None)
                        
        except Exception as e:
            logger.error(f"Shard {self.shard_id}: Error in _on_channel_close for {key}: {e}")
    
    async def get_stub(self, address: str, port: int, stub_type) -> Any:
        """
        获取或创建指定地址、端口和类型的存根
        
        Args:
            address: 目标地址
            port: 目标端口
            stub_type: 存根类型
            
        Returns:
            gRPC存根
        """
        key = (address, port, stub_type)
        
        # 使用信号量控制并发
        acquired = False
        try:
            try:
                acquired = await asyncio.wait_for(self.semaphore.acquire(), timeout=self.lock_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Shard {self.shard_id}: Timeout acquiring semaphore in get_stub for {key}")
                # 超时情况下创建临时通道和存根
                temp_channel = self._create_channel(address, port, cache=False)
                return stub_type(temp_channel)
                
            if acquired:
                try:
                    # 检查缓存
                    if key in self.stubs:
                        return self.stubs[key]
                    
                    # 释放信号量避免递归死锁
                    self.semaphore.release()
                    acquired = False
                    
                    # 获取或创建通道
                    channel = await self.get_channel(address, port)
                    
                    # 重新获取信号量
                    try:
                        acquired = await asyncio.wait_for(self.semaphore.acquire(), timeout=self.lock_timeout)
                    except asyncio.TimeoutError:
                        # 超时直接返回非缓存存根
                        return stub_type(channel)
                    
                    if acquired:
                        # 创建并缓存存根
                        stub = stub_type(channel)
                        self.stubs[key] = stub
                        return stub
                finally:
                    if acquired:
                        self.semaphore.release()
        except Exception as e:
            logger.error(f"Shard {self.shard_id}: Error in get_stub for {key}: {e}")
            
            # 确保释放信号量
            if acquired:
                self.semaphore.release()
                
            # 出错创建临时存根
            temp_channel = self._create_channel(address, port, cache=False)
            return stub_type(temp_channel)
    
    async def with_stub(self, address: str, port: int, stub_type, func, *args, **kwargs):
        """
        使用存根执行函数调用，处理重试逻辑
        
        Args:
            address: 目标地址
            port: 目标端口
            stub_type: 存根类型
            func: 要调用的函数名
            *args, **kwargs: 传递给函数的参数
            
        Returns:
            函数调用结果
        """
        key = (address, port)
        max_retries = 3  # 减少重试次数以避免长时间等待
        retry_count = 0
        last_exception = None
        
        # 熔断器检查
        if key in self.circuit_breakers and not self.circuit_breakers[key].is_allowed():
            raise RuntimeError(f"Circuit breaker open for {key}, service may be unavailable")
        
        while retry_count < max_retries:
            try:
                # 获取存根
                stub = await self.get_stub(address, port, stub_type)
                method = getattr(stub, func)
                
                # 设置超时，避免长时间阻塞
                timeout = 600.0  # 减少默认超时
                
                # 执行RPC调用
                try:
                    result = await asyncio.wait_for(
                        method(*args, **kwargs),
                        timeout=timeout
                    )
                    
                    # 成功时记录到熔断器
                    if key in self.circuit_breakers:
                        self.circuit_breakers[key].record_success()
                        
                    return result
                    
                except asyncio.TimeoutError:
                    error_msg = f"RPC call {func} timed out after {timeout} seconds"
                    logger.warning(f"Shard {self.shard_id}: {error_msg}")
                    raise RuntimeError(error_msg)
                
            except asyncio.TimeoutError:
                # RPC调用超时
                last_exception = RuntimeError(f"RPC call {func} timed out")
                logger.warning(f"Shard {self.shard_id}: Timeout calling {func} on {address}:{port}")
                
                # 记录失败到熔断器
                if key in self.circuit_breakers:
                    self.circuit_breakers[key].record_failure()
                else:
                    self.circuit_breakers[key] = CircuitBreaker()
                    self.circuit_breakers[key].record_failure()
                
                retry_count += 1
                
                # 快速重试，避免长时间等待
                if retry_count < max_retries:
                    wait_time = 0.1 * (2 ** retry_count)  # 指数退避但基数更小
                    await asyncio.sleep(wait_time)
                
            except grpc.aio.AioRpcError as e:
                last_exception = e
                
                # 记录错误细节
                logger.warning(f"Shard {self.shard_id}: gRPC error in {func} on {address}:{port}: {e.code()}: {e.details()}")
                
                # 记录失败到熔断器
                if key in self.circuit_breakers:
                    self.circuit_breakers[key].record_failure()
                else:
                    self.circuit_breakers[key] = CircuitBreaker()
                    self.circuit_breakers[key].record_failure()
                
                # 检查是否应该重试
                retriable_codes = (
                    grpc.StatusCode.UNAVAILABLE, 
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    grpc.StatusCode.CANCELLED,
                    grpc.StatusCode.ABORTED,
                    grpc.StatusCode.UNKNOWN
                )
                
                if e.code() in retriable_codes:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 0.1 * (2 ** retry_count)
                        await asyncio.sleep(wait_time)
                        continue
                
                # 不可重试的错误直接抛出
                logger.error(f"Shard {self.shard_id}: Non-retriable gRPC error: {e.code()}: {e.details()}")
                raise
                
            except Exception as e:
                logger.error(f"Shard {self.shard_id}: Unexpected error in with_stub {address}:{port}: {e}")
                last_exception = e
                
                # 记录失败到熔断器
                if key in self.circuit_breakers:
                    self.circuit_breakers[key].record_failure()
                else:
                    self.circuit_breakers[key] = CircuitBreaker()
                    self.circuit_breakers[key].record_failure()
                
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 0.1 * (2 ** retry_count)
                    await asyncio.sleep(wait_time)
                    continue
                raise
        
        # 重试耗尽
        if last_exception:
            logger.error(f"Shard {self.shard_id}: Failed after {max_retries} attempts: {last_exception}")
            raise last_exception
        else:
            raise RuntimeError(f"Failed to execute {func} after {max_retries} retries")

    async def start_cleanup_task(self):
        """启动清理任务"""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_idle_channels())
            logger.debug(f"Shard {self.shard_id}: Cleanup task started")
    
    async def _cleanup_idle_channels(self):
        """定期清理空闲通道"""
        try:
            while True:
                await asyncio.sleep(300)  # 降低清理频率到5分钟一次
                
                # 不使用信号量锁定太久，而是创建快照后处理
                to_check = []
                
                async with self.semaphore:
                    current_time = time.time()
                    for key, (channel, last_used, _) in self.channels.items():
                        if current_time - last_used > self.max_idle_time:
                            to_check.append((key, channel))
                
                # 在锁外逐个关闭通道
                for key, channel in to_check:
                    try:
                        # 再次检查是否仍需要关闭
                        should_close = False
                        async with self.semaphore:
                            if key in self.channels:
                                _, last_used, _ = self.channels[key]
                                if time.time() - last_used > self.max_idle_time:
                                    self.channels.pop(key)
                                    should_close = True
                                    
                                    # 移除相关存根
                                    stub_keys = [k for k in self.stubs.keys() if k[0] == key[0] and k[1] == key[1]]
                                    for stub_key in stub_keys:
                                        self.stubs.pop(stub_key, None)
                                    
                                    # 从已知良好连接集合中移除
                                    if key in self.known_good_connections:
                                        self.known_good_connections.remove(key)
                        
                        # 如果确定要关闭，在锁外执行
                        if should_close:
                            await channel.close()
                    except Exception as e:
                        logger.warning(f"Shard {self.shard_id}: Error closing idle channel for {key}: {e}")
                        
        except asyncio.CancelledError:
            logger.debug(f"Shard {self.shard_id}: Cleanup task cancelled")
        except Exception as e:
            logger.error(f"Shard {self.shard_id}: Error in cleanup task: {e}")
            
    async def preconnect(self, workers: List[Tuple[str, int]]):
        """
        预连接到所有工作节点
        
        Args:
            workers: 工作节点列表，每项为(地址, 端口)元组
        """
        if self.preconnected:
            return
            
        logger.info(f"Shard {self.shard_id}: Pre-connecting to {len(workers)} workers")
        
        # 使用信号量限制并发连接数
        sem = asyncio.Semaphore(min(10, len(workers)))
        
        async def connect_one(addr, port):
            async with sem:
                try:
                    channel = await self.get_channel(addr, port)
                    return (addr, port), True
                except Exception as e:
                    logger.warning(f"Shard {self.shard_id}: Failed to pre-connect to {addr}:{port}: {e}")
                    return (addr, port), False
        
        # 创建所有连接任务
        tasks = [connect_one(addr, port) for addr, port in workers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 分析结果
        success_count = sum(1 for result in results if isinstance(result, tuple) and result[1])
        logger.info(f"Shard {self.shard_id}: Successfully pre-connected to {success_count}/{len(workers)} workers")
        
        self.preconnected = True


class ShardedConnectionManager:
    """
    分片连接管理器，通过哈希分片减少锁竞争
    """
    
    _instance = None
    _init_lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ShardedConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # 配置
        self.shard_count = 32 # 分片数量
        self.semaphore_size = 10 # 每个分片的信号量大小
        self.lock_timeout = 5.0  # 锁超时时间 (Reduced from 30.0)
        self.max_idle_time = 600.0  # 空闲通道超时时间 (10 minutes)
        
        # 创建分片
        self.shards = [
            ConnectionShard(
                shard_id=i, 
                semaphore_size=self.semaphore_size,
                lock_timeout=self.lock_timeout,
                max_idle_time=self.max_idle_time
            ) 
            for i in range(self.shard_count)
        ]
        
        # 预连接状态
        self.preconnected = False
        self._initialized = True
    
    def get_shard(self, address: str, port: int) -> ConnectionShard:
        """
        根据地址和端口选择对应的分片
        
        Args:
            address: 目标地址
            port: 目标端口
            
        Returns:
            对应的连接分片
        """
        # 使用一致性哈希避免热点
        key = f"{address}:{port}"
        shard_index = hash(key) % self.shard_count
        return self.shards[shard_index]
    
    async def initialize(self):
        """初始化连接管理器（只需执行一次）"""
        async with self._init_lock:
            # 启动所有分片的清理任务
            for shard in self.shards:
                await shard.start_cleanup_task()
            
            logger.info(f"Initialized connection manager with {self.shard_count} shards")
    
    async def get_channel(self, address: str, port: int) -> grpc.aio.Channel:
        """
        获取或创建通道
        
        Args:
            address: 目标地址
            port: 目标端口
            
        Returns:
            gRPC异步通道
        """
        shard = self.get_shard(address, port)
        return await shard.get_channel(address, port)
    
    async def get_stub(self, address: str, port: int, stub_type) -> Any:
        """
        获取或创建存根
        
        Args:
            address: 目标地址
            port: 目标端口
            stub_type: 存根类型
            
        Returns:
            gRPC存根
        """
        shard = self.get_shard(address, port)
        return await shard.get_stub(address, port, stub_type)
    
    async def with_stub(self, address: str, port: int, stub_type, func, *args, **kwargs):
        """
        使用存根执行函数调用
        
        Args:
            address: 目标地址
            port: 目标端口
            stub_type: 存根类型
            func: 要调用的函数名
            *args, **kwargs: 传递给函数的参数
            
        Returns:
            函数调用结果
        """
        shard = self.get_shard(address, port)
        return await shard.with_stub(address, port, stub_type, func, *args, **kwargs)
    
    async def preconnect(self, workers: List[Tuple[str, int]]):
        """
        预连接到所有工作节点
        
        Args:
            workers: 工作节点列表
        """
        if self.preconnected:
            return
            
        logger.info(f"Pre-connecting to {len(workers)} workers across {self.shard_count} shards")
        
        # 将工作节点分配到各个分片
        shard_workers = [[] for _ in range(self.shard_count)]
        for addr, port in workers:
            shard_index = hash(f"{addr}:{port}") % self.shard_count
            shard_workers[shard_index].append((addr, port))
        
        # 并行执行各分片的预连接
        tasks = []
        for i, worker_list in enumerate(shard_workers):
            if worker_list:
                tasks.append(self.shards[i].preconnect(worker_list))
        
        await asyncio.gather(*tasks)
        self.preconnected = True
        logger.info("Pre-connection to all workers completed")
    
    async def stop(self):
        """停止连接管理器，关闭所有通道"""
        logger.info("Stopping connection manager")
        
        # 收集所有通道以关闭
        all_channels = []
        for shard in self.shards:
            async with shard.semaphore:
                for key, (channel, _, _) in list(shard.channels.items()):
                    all_channels.append((key, channel))
                shard.channels.clear()
                shard.stubs.clear()
                shard.known_good_connections.clear()
                
                if shard.cleanup_task and not shard.cleanup_task.done():
                    shard.cleanup_task.cancel()
        
        # 关闭所有通道（不使用锁）
        for key, channel in all_channels:
            try:
                await asyncio.wait_for(channel.close(), timeout=1.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Error closing channel for {key}: {e}")
        
        logger.info("Connection manager stopped")
    
    def sync_stop(self):
        """同步停止方法，用于非异步上下文"""
        logger.info("Sync stopping connection manager")
        for shard in self.shards:
            if shard.cleanup_task and not shard.cleanup_task.done():
                shard.cleanup_task.cancel()


# 创建全局单例
connection_manager = ShardedConnectionManager()


# 为全局初始化提供便捷方法
async def initialize_connection_manager(workers: List[Tuple[str, int]] = None):
    """
    全局初始化连接管理器，可选预连接到工作节点
    
    Args:
        workers: 可选的工作节点列表，如果提供则预连接
    """
    await connection_manager.initialize()
    
    if workers:
        await connection_manager.preconnect(workers)

    return connection_manager