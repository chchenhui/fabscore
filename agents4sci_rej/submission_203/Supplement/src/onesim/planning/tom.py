from onesim.planning.base import PlanningBase
from onesim.agent.general_agent import Message

class TOMPlanning(PlanningBase):
    def __init__(self,model_config_name,sys_prompt):
        super().__init__(model_config_name,sys_prompt)
    
    async def plan(self, **kwargs) -> str:
        prompt = f"""
        ### Observation
        {kwargs["observation"]}
        
        ### Instruction
        {kwargs["instruction"]}
        
        ### Relationship
        {kwargs["relationship"]}

        Analyze the mental states of other agents in this scenario:
        
        1. What are other agents likely thinking or believing?
        
        2. What might be their intentions and goals?
        
        3. How might they react to different actions?
        
        Please think based on the above concisely.
        """
        
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt, role="user")
        )
        response = await self.model.acall(prompt)
        return response.text