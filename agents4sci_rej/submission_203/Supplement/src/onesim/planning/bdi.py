from onesim.planning.base import PlanningBase
from onesim.agent.general_agent import Message

class BDIPlanning(PlanningBase):
    def __init__(self,model_config_name,sys_prompt):
        super().__init__(model_config_name,sys_prompt)
    
    async def plan(self, **kwargs) -> str:
        prompt = f"""
        ### Agent Profile
        {kwargs["profile"]}

        ### Memory (Beliefs)
        {kwargs["memory"]}
        
        ### Observation (New Beliefs)
        {kwargs["observation"]}
        
        ### Instruction (Task)
        {kwargs["instruction"]}

        Please analyze the situation using the BDI (Belief-Desire-Intention) framework:
        
        1. Beliefs: Based on the agent's memory and current observations, what does the agent believe about the current state of the world?
        
        2. Desires: Given these beliefs and the agent's profile, what goals should the agent prioritize?
        
        3. Intentions: What specific actions should the agent commit to in order to achieve these goals?
        
        Please think based on the above concisely.
        """
        
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt, role="user")
        )
        response = await self.model.acall(prompt)
        return response.text