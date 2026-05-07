from .inspiration_agent import InspirationAgent
from .detailer_agent import DetailerAgent
from .experiments_agent import Experimentagent
from .intervention_agent import InterventionAgent
from loguru import logger


class Paradigm4Agent:
    """
    Manages all agents for Paradigm A.
    """
    def __init__(self, model_name):
        self.inspiration_agent = InspirationAgent(model_name)
        self.detailer_agent = DetailerAgent(model_name)
        self.experiment_agent = Experimentagent(model_name)
        self.intervention_agent = InterventionAgent(model_name)


    def run(self, user_input):
        print(user_input)
        """
        Executes the agents in a defined sequence or collaboration.
        """
        logger.info("\n[Step 1/4] Generating scenarios...")
        inspiration_input = {"theory": user_input.get("theory"), "observation": user_input.get("observation"), "condition": user_input.get("condition")}
        inspiration_output = self.inspiration_agent.process(inspiration_input)

        logger.info("\n[Step 2/4] Elaborating the scenario into an ODD protocol......")
        odd_output = self.detailer_agent.process(inspiration_output)

        print("=" * 80)
        print(odd_output)
        print("=" * 80)
        logger.info("\n[Step 3/4] Designing experiments...")
        experiment_input = {
            "scene_information": odd_output.get("odd_protocol", ""),
            "scenario": inspiration_output.get("scenario", ""),
            "scene_name": user_input.get("scene_name", "")
        }
        experiment_output = self.experiment_agent.process(experiment_input)
        intervention_input = {
            "scene_information": odd_output.get("odd_protocol", ""),
            "scenario": inspiration_output.get("scenario", ""),
            "exp_config": experiment_output,
        }
        logger.info("\n[Step 4/4] Designing interventions...")
        intervention_output = self.intervention_agent.process(intervention_input)

        return inspiration_output, odd_output, experiment_output, intervention_output