from onesim.events import Event
from typing import Dict
class StartEvent(Event):
    """环境启动事件"""
    def __init__(self, from_agent_id: str, to_agent_id: str):
        super().__init__(from_agent_id, to_agent_id)

# 更新CulturalRecommendationEvent以传递完整的文化特征集
class RecommendationEvent(Event):
    def __init__(self, from_agent_id: str, to_agent_id: str, cultural_traits: Dict[str, int], 
                    reason: str):
        super().__init__(from_agent_id, to_agent_id)  # emoji参数留空
        self.reason = reason  # 推荐理由
        self.cultural_traits = cultural_traits  # 完整的文化特征集

class AdoptionEvent(Event):
    def __init__(self, from_agent_id: str, to_agent_id: str, 
                 dimension: str, old_value: str, new_value: str, 
                  adopted: bool, round_number: int):
        super().__init__(from_agent_id, to_agent_id)  # emoji参数留空
        self.adopted = adopted  # 是否被采纳
        self.round_number = round_number  # 当前轮次
        self.dimension = dimension  # 文化维度
        self.old_value = old_value  # 旧文化特征值
        self.new_value = new_value  # 新文化特征值
