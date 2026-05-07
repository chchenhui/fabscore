from typing import Any, List, Optional, Dict, Tuple
import asyncio
import random
from loguru import logger
from onesim.agent import GeneralAgent
from onesim.profile import AgentProfile
from onesim.memory import MemoryStrategy
from onesim.planning import PlanningBase
from onesim.events import Event
from onesim.relationship import RelationshipManager
from .events import RecommendationEvent, AdoptionEvent

class CulturalAgent(GeneralAgent):
    """基于文化传播模型的Agent，由LLM做出文化交流决策"""
    
    # 文化维度列表
    CULTURAL_DIMENSIONS = [
        "music_preference",
        "culinary_preference", 
        "fashion_style", 
        "political_orientation", 
        "leisure_activity"
    ]
    
    def __init__(self,
                 sys_prompt: str | None = None,
                 model_config_name: str = None,
                 event_bus_queue: asyncio.Queue = None,
                 profile: AgentProfile = None,
                 memory: MemoryStrategy = None,
                 planning: PlanningBase = None,
                 relationship_manager: RelationshipManager = None) -> None:
        
        # 使用定制的系统提示如果没有提供
        if sys_prompt is None:
            sys_prompt = self.create_cultural_agent_sys_prompt()
            
        super().__init__(sys_prompt, model_config_name, event_bus_queue, profile, memory, planning, relationship_manager)
        # 注册两个主要事件处理器：发送推荐和处理推荐
        self.register_event("StartEvent", "send_recommendation")
        self.register_event("RecommendationEvent", "receive_recommendation")
        
        # 初始化文化解释（如果还没有）
        if not self.profile.get_data("trait_explanations"):
            self.initialize_trait_explanations()

    def create_cultural_agent_sys_prompt(self):
        """创建简化但有效的系统提示"""
        return """
You have cultural traits across five dimensions (music, food, fashion, politics, leisure).

- Agents who share some cultural traits but differ in others are most open to cultural exchange
- Moderate cultural similarity creates the best foundation for meaningful communication and sharing
- When cultures are too similar or too different, there's little motivation for cultural exchange
- You are naturally curious about others' cultural practices and enjoy sharing your own experiences

When making decisions about cultural interactions, actively seek opportunities for meaningful dialogue and cultural exchange while staying true to your core values.
"""

    def get_cultural_traits(self) -> Dict[str, str]:
        """获取Agent的所有文化特征"""
        traits = {}
        for dimension in self.CULTURAL_DIMENSIONS:
            traits[dimension] = self.profile.get_data(dimension, "")
        return traits

    def initialize_trait_explanations(self):
        """为文化特征生成简单解释"""
        traits = self.get_cultural_traits()
        explanations = {}
        for dimension, trait_value in traits.items():
            if trait_value:  # 只为有值的特征生成解释
                explanations[dimension] = f"I value {trait_value} because it aligns with my personal preferences."
        
        self.profile.update_data("trait_explanations", explanations)

    def calculate_similarity(self, other_traits: Dict[str, str]) -> float:
        """计算与另一个Agent的文化相似度"""
        my_traits = self.get_cultural_traits()
        if not my_traits or not other_traits:
            return 0.0
        
        # 创建一个包含所有维度的集合
        all_dimensions = set(my_traits.keys()).union(other_traits.keys())
        
        # 计算相同特征的数量
        shared_count = 0
        for dim in all_dimensions:
            if dim in my_traits and dim in other_traits and my_traits[dim] == other_traits[dim]:
                shared_count += 1
        
        # 相似度是共享相同值的维度比例
        return shared_count / len(all_dimensions) if all_dimensions else 0.0

    def find_different_dimensions(self, other_traits: Dict[str, str]) -> List[Dict[str, str]]:
        """找出与另一个Agent不同的文化维度及其值"""
        my_traits = self.get_cultural_traits()
        different_dims = []
        
        for dim in self.CULTURAL_DIMENSIONS:
            if dim in my_traits and dim in other_traits and my_traits[dim] != other_traits[dim]:
                different_dims.append({
                    "dimension": dim,
                    "current_value": my_traits[dim],
                    "other_value": other_traits[dim]
                })
                
        return different_dims

    def adjust_target_preferences_by_openness(self, potential_targets: List[Dict]) -> List[Dict]:
        """根据开放性水平调整目标偏好权重"""
        openness_level = self.profile.get_data("openness_level", "Moderate openness")
        
        for target in potential_targets:
            similarity = target["similarity"]
            base_probability = target["interaction_probability"]
            
            # 根据开放性调整偏好
            if openness_level == "Very low openness":
                # 强烈偏好高相似度(0.7-1.0)
                if similarity >= 0.7:
                    target["openness_adjusted_probability"] = base_probability * 2.0
                else:
                    target["openness_adjusted_probability"] = base_probability * 0.2
            elif openness_level == "Low openness":
                # 偏好中高相似度(0.6-0.8)
                if 0.6 <= similarity <= 0.8:
                    target["openness_adjusted_probability"] = base_probability * 1.5
                else:
                    target["openness_adjusted_probability"] = base_probability * 0.5
            elif openness_level == "Moderate openness":
                # 保持原有偏好(0.4-0.7)
                target["openness_adjusted_probability"] = base_probability
            elif openness_level == "High openness":
                # 偏好中低相似度(0.3-0.6)
                if 0.3 <= similarity <= 0.6:
                    target["openness_adjusted_probability"] = base_probability * 1.5
                else:
                    target["openness_adjusted_probability"] = base_probability * 0.7
            elif openness_level == "Very high openness":
                # 强烈偏好低相似度(0.0-0.4)
                if similarity <= 0.4:
                    target["openness_adjusted_probability"] = base_probability * 2.0
                else:
                    target["openness_adjusted_probability"] = base_probability * 0.3
            else:
                target["openness_adjusted_probability"] = base_probability
                
        return potential_targets

    def should_adopt_based_on_openness(self, similarity: float) -> float:
        """根据开放性水平和相似度计算采纳倾向概率（0~0.95）"""
        openness_level = self.profile.get_data("openness_level", "Moderate openness")
        
        # 基础采纳概率（基于相似度的理论概率）
        base_adoption_probability = 4 * similarity * (1 - similarity)
        
        # 根据开放性调整采纳倾向
        if openness_level == "Very low openness":
            # 非常保守，降低采纳概率
            adjusted_probability = base_adoption_probability * 0.3
        elif openness_level == "Low openness":
            # 较为保守，略微降低采纳概率
            adjusted_probability = base_adoption_probability * 0.6
        elif openness_level == "Moderate openness":
            # 保持基础概率
            adjusted_probability = base_adoption_probability
        elif openness_level == "High openness":
            # 较为开放，提高采纳概率
            adjusted_probability = base_adoption_probability * 1.4
        elif openness_level == "Very high openness":
            # 非常开放，大幅提高采纳概率
            adjusted_probability = base_adoption_probability * 1.8
        else:
            adjusted_probability = base_adoption_probability
        
        # 确保概率在合理范围内
        adjusted_probability = min(adjusted_probability, 0.95)
        
        # 返回概率值供LLM作为决策参考
        return adjusted_probability

    async def send_recommendation(self, event: Event) -> List[Event]:
        """向社交网络中的一个Agent发送文化特征集，让LLM选择对象"""
        
        traits = self.get_cultural_traits()
        if not traits:
            logger.warning(f"Agent {self.profile_id} doesn't have cultural traits")
            return []

        # 获取社交网络信息
        relationships = self.relationship_manager.get_all_relationships()
        potential_targets = []
        
        for relation in relationships:
            if relation.target_id == "ENV":
                continue
                
            target_info = relation.get_target_info()
            
            # 构建目标的文化特征
            target_traits = {}
            for dimension in self.CULTURAL_DIMENSIONS:
                if dimension in target_info:
                    target_traits[dimension] = target_info.get(dimension, "")
            
            similarity = self.calculate_similarity(target_traits)
            
            # 计算对应的理论互动概率
            interaction_probability = 4 * similarity * (1 - similarity)
            
            # 获取目标名称
            target_name = target_info.get("name", relation.target_id)
            
            # 收集信息
            potential_targets.append({
                "id": relation.target_id,
                "name": target_name,
                "similarity": similarity,
                "interaction_probability": interaction_probability,
                "traits": target_traits
            })
        
        if not potential_targets:
            logger.warning(f"Agent {self.profile_id} has no potential targets")
            return []
        
        # 根据开放性水平调整目标偏好
        potential_targets = self.adjust_target_preferences_by_openness(potential_targets)
            
        # 让LLM选择互动对象
        instruction = """
        Choose ONE agent from your social network to engage in cultural exchange and sharing.
        
        Use these principles for meaningful cultural interaction:
        - Moderate similarity generally offers the best opportunities for mutual learning.
        - Avoid extremes: if similarity is ~0.0 or ~1.0, meaningful exchange is unlikely.
        - Consider your openness level as a soft preference, not a hard rule:
          • Very low openness → lean toward higher similarity (≥0.6) while avoiding near-1.0.
          • Low openness → prefer moderately high similarity (0.6–0.8).
          • Moderate openness → prefer moderate similarity (0.4–0.7).
          • High openness → accept 0.3–0.7, with a mild preference for 0.4–0.6.
          • Very high openness → accept 0.2–0.8, with a gentle preference for 0.3–0.6; avoid extremes (<0.2 or >0.9) unless you can justify strong common ground.
        
        Select a target for cultural dialogue and explain why you chose them.
        
        Return your decision in this JSON format:
        {
            "target_id": "<ID of the chosen agent>",
            "reason": "<Brief explanation for your choice (1-2 sentences)>"
        }
        """
        
        # 构建观察信息，包含自己的文化特征和潜在目标的信息
        my_profile = ", ".join([f"{dim.replace('_', ' ')}: {val}" for dim, val in traits.items() if val])
        
        # 获取自己的开放性水平
        openness_level = self.profile.get_data("openness_level", "Moderate openness")
        
        # 格式化潜在目标信息
        targets_info = ""
        for target in potential_targets:
            target_profile = ", ".join([f"{dim.replace('_', ' ')}: {val}" for dim, val in target["traits"].items() if val])
            openness_prob = target.get("openness_adjusted_probability", target["interaction_probability"])
            targets_info += f"- ID: {target['id']}, Similarity: {target['similarity']:.2f}, Base probability: {target['interaction_probability']:.2f}, Openness-adjusted probability: {openness_prob:.2f}\n  Cultural traits: {target_profile}\n\n"
        
        observation = f"""
        YOUR CULTURAL PROFILE:
        {my_profile}
        Your openness level: {openness_level}
        
        POTENTIAL INTERACTION TARGETS:
        {targets_info}
        """
        
        result = await self.generate_reaction(instruction, observation)
        
        # 获取LLM选择的目标
        target_id = result.get('target_id')
        reason = result.get('reason', "This agent seems most receptible to cultural influence.")
        
        # 验证目标是否有效
        if not target_id or not any(t["id"] == target_id for t in potential_targets):
            logger.warning(f"Agent {self.profile_id} selected invalid target: {target_id}")
            # 随机选择一个有效目标
            if potential_targets:
                target = random.choice(potential_targets)
                target_id = target["id"]
                logger.info(f"Falling back to random target: {target_id}")
            else:
                return []
        
        # 获取目标特征和相似度
        target_traits = {}
        similarity = 0.0
        for t in potential_targets:
            if t["id"] == target_id:
                target_traits = t["traits"]
                similarity = t["similarity"]
                break
        
        # 记录这次互动
        recommendation_history = self.profile.get_data("recommendation_history", [])
        recommendation_history.append({
            "round": await self.env.get_data("round_number", 0),
            "target_id": target_id,
            "similarity": similarity,
            "reason": reason
        })
        if len(recommendation_history)>10:
            recommendation_history=recommendation_history[-10:]
        self.profile.update_data("recommendation_history", recommendation_history)
        
        # 创建推荐事件，发送完整的文化特征集
        recommendation_event = RecommendationEvent(
            from_agent_id=self.profile_id,
            to_agent_id=target_id,
            cultural_traits=traits,
            reason=reason
        )
        
        # logger.info(f"Agent {self.profile_id} sends cultural traits to Agent {target_id} (similarity: {similarity:.2f})")
        return [recommendation_event]

    async def receive_recommendation(self, event: Event) -> List[Event]:
        """
        接收文化特征集并由LLM决定是否采纳其中的特征
        """

        recommender_id = event.from_agent_id
        recommender_traits = event.cultural_traits
        reason = event.reason
        
        my_traits = self.get_cultural_traits()
        
        # 计算文化相似度
        similarity = self.calculate_similarity(recommender_traits)
        
        # 如果相似度为0或1，根据文化传播理论不会发生文化交流
        if similarity == 0.0 or similarity == 1.0:
            logger.info(f"Agent {self.profile_id} and {recommender_id} have similarity {similarity}, no transmission")
            
            # 记录不发生传播的情况
            adoption_history = self.profile.get_data("adoption_history", [])
            adoption_history.append({
                "round": await self.env.get_data("round_number", 0),
                "recommender": recommender_id,
                "similarity": similarity,
                "adopted": False,
                "reasoning": f"Similarity is {similarity}, no meaningful cultural exchange occurs"
            })
            self.profile.update_data("adoption_history", adoption_history)
            
            # 创建拒绝采纳事件
            adoption_event = AdoptionEvent(
                from_agent_id=self.profile_id,
                to_agent_id="ENV",
                dimension="",
                old_value="",
                new_value="",
                adopted=False,
                round_number=await self.env.get_data("round_number", 0)
            )
            
            return [adoption_event]
        
        # 找出不同的维度
        different_dimensions = self.find_different_dimensions(recommender_traits)
        
        # 如果没有不同的维度，不会发生传播
        if not different_dimensions:
            logger.warning(f"Agent {self.profile_id} and {recommender_id} have no different dimensions despite similarity {similarity}")
            
            # 创建拒绝采纳事件
            adoption_event = AdoptionEvent(
                from_agent_id=self.profile_id,
                to_agent_id="ENV",
                dimension="",
                old_value="",
                new_value="",
                adopted=False,
                round_number=await self.env.get_data("round_number", 0)
            )
            
            return [adoption_event]
        
        # 让LLM选择要采纳的维度
        instruction = """
        You've engaged in cultural exchange with another agent and learned about their cultural practices.
        
        Through meaningful dialogue and mutual understanding, you may embrace new cultural elements that resonate with you — or you may decide not to adopt at this time.
        Consider:
        - Your adoption tendency (a probability derived from your openness and similarity) is guidance, not a hard rule.
        - Avoid adopting from extreme similarity (~0.0 or ~1.0), as meaningful exchange is unlikely.
        - Prefer adoptions where you can articulate a clear personal value or meaningful enrichment.
        
        From the cultural differences you've discovered, decide whether to adopt ONE dimension. If you choose not to adopt, state your reason.
        
        Return your decision in this JSON format:
        {
            "adopt": true | false,
            "adopt_dimension": "<dimension_name if adopt=true, else empty string>",
            "reasoning": "<Brief explanation for your choice (1-2 sentences)>"
        }
        """
        
        # 构建观察信息
        my_profile = ", ".join([f"{dim.replace('_', ' ')}: {val}" for dim, val in my_traits.items() if val])
        recommender_profile = ", ".join([f"{dim.replace('_', ' ')}: {val}" for dim, val in recommender_traits.items() if val])
        
        # 获取个性特质和开放性水平
        personality = self.profile.get_data("personality_trait", "neutral")
        openness_level = self.profile.get_data("openness_level", "Moderate openness")
        
        # 计算基于开放性的采纳倾向
        adoption_tendency = self.should_adopt_based_on_openness(similarity)
        
        # 格式化不同的维度
        differences_info = ""
        for diff in different_dimensions:
            differences_info += f"- {diff['dimension'].replace('_', ' ')}:\n"
            differences_info += f"  Your current: {diff['current_value']}\n"
            differences_info += f"  Their value: {diff['other_value']}\n\n"
        
        observation = f"""
        CULTURAL EXCHANGE INFORMATION:
        Recommender: Agent {recommender_id}
        Similarity: {similarity:.2f}
        Your personality: {personality}
        Your openness level: {openness_level}
        Your adoption tendency (based on openness): {adoption_tendency:.2f}
        Recommender's explanation: "{reason}"
        
        RECOMMENDER'S CULTURAL PROFILE:
        {recommender_profile}
        
        CULTURAL DIFFERENCES (potential to adopt):
        {differences_info}
        """
        
        result = await self.generate_reaction(instruction, observation)
        
        # LLM 是否选择采纳
        adopt_flag = bool(result.get('adopt', self.should_adopt_based_on_openness(similarity) >= 0.5))
        adopt_dimension = result.get('adopt_dimension', '').lower().replace(' ', '_')
        reasoning = result.get('reasoning', "This cultural trait seems valuable to adopt.")

        if not adopt_flag:
            # 记录未采纳
            adoption_history = self.profile.get_data("adoption_history", [])
            adoption_history.append({
                "round": await self.env.get_data("round_number", 0),
                "recommender": recommender_id,
                "similarity": similarity,
                "adopted": False,
                "reasoning": reasoning
            })
            if len(adoption_history)>10:
                adoption_history=adoption_history[-10:]
            self.profile.update_data("adoption_history", adoption_history)

            adoption_event = AdoptionEvent(
                from_agent_id=self.profile_id,
                to_agent_id="ENV",
                dimension="",
                old_value="",
                new_value="",
                adopted=False,
                round_number=await self.env.get_data("round_number", 0)
            )

            return [adoption_event]
        
        # 验证维度是否有效
        valid_dimension = False
        dimension_info = None
        for diff in different_dimensions:
            if diff['dimension'] == adopt_dimension:
                valid_dimension = True
                dimension_info = diff
                break
        
        # 如果维度无效，随机选择一个
        if not valid_dimension:
            logger.warning(f"Agent {self.profile_id} selected invalid dimension: {adopt_dimension}")
            dimension_info = random.choice(different_dimensions)
            adopt_dimension = dimension_info['dimension']
            logger.info(f"Falling back to random dimension: {adopt_dimension}")
        
        # 提取旧值和新值
        old_value = dimension_info['current_value']
        new_value = dimension_info['other_value']
        
        # 更新文化特征
        self.profile.update_data(adopt_dimension, new_value)
        
        # 更新解释
        trait_explanations = self.profile.get_data("trait_explanations", {})
        trait_explanations[adopt_dimension] = reasoning
        self.profile.update_data("trait_explanations", trait_explanations)
        
        # 记录变化
        adoption_history = self.profile.get_data("adoption_history", [])
        adoption_history.append({
            "round": await self.env.get_data("round_number", 0),
            "dimension": adopt_dimension,
            "old_value": old_value,
            "new_value": new_value,
            "recommender": recommender_id,
            "similarity": similarity,
            "adopted": True,
            "reasoning": reasoning
        })
        if len(adoption_history)>10:
            adoption_history=adoption_history[-10:]
        self.profile.update_data("adoption_history", adoption_history)
        
        #logger.info(f"Agent {self.profile_id} adopted {adopt_dimension}: {new_value} from {recommender_id}")
        
        # 创建采纳事件通知环境
        adoption_event = AdoptionEvent(
            from_agent_id=self.profile_id,
            to_agent_id="ENV",
            dimension=adopt_dimension,
            old_value=old_value,
            new_value=new_value,
            adopted=True,
            round_number=await self.env.get_data("round_number", 0)
        )
        
        return [adoption_event]
