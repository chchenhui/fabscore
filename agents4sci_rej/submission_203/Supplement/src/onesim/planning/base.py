from onesim.models import ModelManager
class PlanningBase:
    def __init__(self,model_config_name,sys_prompt):
        model_manager = ModelManager.get_instance()
        self.model= model_manager.get_model(
                model_config_name=model_config_name,
            )
        self.sys_prompt=sys_prompt

    async def plan(self, **kwargs) -> str:
        pass
