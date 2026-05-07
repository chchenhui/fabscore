from onesim.planning.base import PlanningBase
from onesim.agent.general_agent import Message

class COTPlanning(PlanningBase):
    def __init__(self,model_config_name,sys_prompt):
        super().__init__(model_config_name,sys_prompt)

    async def plan(self,**kwargs) -> str:
        prompt=f"""
        ### Agent Profile
        {kwargs["profile"]}

        ### Memory
        {kwargs["memory"]}

        
        ### Observation
        {kwargs["observation"]}
        
        ### Instruction
        {kwargs["instruction"]}

        Please think step by step based on the above concisely.
        """
        prompt=self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt, role="user")
        )
        response = await self.model.acall(prompt)
        return response.text