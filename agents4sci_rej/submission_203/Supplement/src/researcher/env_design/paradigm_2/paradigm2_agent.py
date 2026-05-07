from .inspiration_agent import InspirationAgent
from .detailer_agent import DetailerAgent
from loguru import logger


class Paradigm2Agent:
    """
    Manages all agents for Paradigm A.
    """
    def __init__(self, model_name):
        self.inspiration_agent = InspirationAgent(model_name)
        self.detailer_agent = DetailerAgent(model_name)

    def run(self, user_input):
        print(user_input)
        """
        Executes the agents in a defined sequence or collaboration.
        """
        logger.info("\n[Step 1/2] Generating potential research questions and scenarios...")
        inspiration_input = {"observation": user_input.get("observation"), "condition": user_input.get("condition")}
        inspiration_output = self.inspiration_agent.process(inspiration_input)

        logger.info("\n[Step 2/2] Elaborating the scenario into an ODD protocol......")
        odd_output = self.detailer_agent.process(inspiration_output)
        
        return inspiration_output, odd_output