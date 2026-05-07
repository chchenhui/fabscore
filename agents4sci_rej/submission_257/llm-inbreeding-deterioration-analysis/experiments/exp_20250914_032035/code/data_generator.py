"""
Data Generator for LLM Inbreeding Deterioration Analysis

This module generates and manages datasets for multi-generation training experiments.
It creates synthetic text tasks and manages data flow between generations.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiGenerationDataGenerator:
    """Generates and manages datasets for multi-generation training experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"])
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.seed = config["random_seed"]
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Create data directory
        self.data_dir = Path(config["paths"]["data_dir"])
        self.data_dir.mkdir(exist_ok=True)
        
    def generate_base_human_data(self) -> DatasetDict:
        """Generate base human-like training data for the first generation."""
        logger.info("Generating base human training data...")
        
        # Task templates for different types of text generation
        task_templates = {
            "question_answering": {
                "questions": [
                    "What is the capital of France?",
                    "How do photosynthesis work?",
                    "Explain the concept of gravity.",
                    "What causes the seasons?",
                    "How do computers process information?",
                    "What is DNA and what does it do?",
                    "Explain how the internet works.",
                    "What is climate change?",
                    "How do vaccines work?",
                    "What is artificial intelligence?"
                ],
                "answers": [
                    "The capital of France is Paris, a historic city known for its culture and landmarks.",
                    "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
                    "Gravity is a fundamental force that attracts objects with mass toward each other, giving weight to physical objects.",
                    "Seasons are caused by Earth's axial tilt as it orbits the sun, creating varying amounts of daylight and solar heating.",
                    "Computers process information using binary code, transistors, and logical operations to perform calculations and execute programs.",
                    "DNA is the genetic material that contains instructions for building and maintaining living organisms.",
                    "The internet is a global network of interconnected computers that share information using standardized protocols.",
                    "Climate change refers to long-term changes in global temperature and weather patterns, largely due to human activities.",
                    "Vaccines work by training the immune system to recognize and fight specific diseases without causing illness.",
                    "Artificial intelligence is the development of computer systems that can perform tasks typically requiring human intelligence."
                ]
            },
            
            "summarization": {
                "texts": [
                    "The Industrial Revolution was a period of major technological change that began in Britain in the late 18th century. It involved the transition from hand production methods to machines, new chemical manufacturing processes, and improved transportation systems. This transformation spread to Western Europe and North America, fundamentally changing society, economics, and daily life.",
                    "Renewable energy sources like solar, wind, and hydroelectric power are becoming increasingly important as alternatives to fossil fuels. These sources offer environmental benefits by reducing greenhouse gas emissions and provide sustainable options for meeting growing energy demands. Advances in technology have made renewable energy more efficient and cost-effective.",
                    "Machine learning is a subset of artificial intelligence that enables computers to learn and improve performance without explicit programming. It uses algorithms to analyze data, identify patterns, and make predictions or decisions. Applications include image recognition, natural language processing, and recommendation systems."
                ],
                "summaries": [
                    "The Industrial Revolution began in 18th century Britain, introducing mechanization and new manufacturing processes that transformed society.",
                    "Renewable energy sources offer environmental benefits and sustainable alternatives to fossil fuels, with improving technology making them more viable.",
                    "Machine learning enables computers to learn from data and improve performance automatically, with applications in various fields."
                ]
            },
            
            "creative_writing": {
                "prompts": [
                    "Write a short story about a time traveler who discovers something unexpected.",
                    "Describe a futuristic city where nature and technology coexist perfectly.",
                    "Create a dialogue between two characters meeting for the first time.",
                    "Write about a day when all the books in the world suddenly disappeared.",
                    "Describe an alien planet from the perspective of an explorer."
                ],
                "responses": [
                    "Sarah stepped out of her time machine into 1823, expecting to find a bustling London street. Instead, she found herself in a quiet meadow filled with flowers that wouldn't be discovered by botanists for another century.",
                    "Neo-Eden stretched before her, its crystalline towers intertwined with massive trees whose bioluminescent leaves powered the city. Gardens cascaded down building walls while birds nested in architectural alcoves designed specifically for them.",
                    "'I've never seen anyone read a physical book on this train before,' Maria said to the stranger across from her. 'Oh, this?' he replied, holding up the worn novel. 'It belonged to my grandmother. I find that paper holds memories better than screens.'",
                    "When Maya woke up, every bookshelf in her apartment was empty. Not just empty—the books had vanished entirely, leaving not even dust outlines. Outside, people wandered the streets in confusion, unable to remember what used to fill those mysterious rectangular spaces.",
                    "The twin suns cast purple shadows across the crystalline landscape of Xerion Prime. Each step produced a musical note as my boots struck the resonant ground, creating an otherworldly symphony that echoed across the valleys."
                ]
            }
        }
        
        # Generate dataset
        train_data = []
        val_data = []
        test_data = []
        
        for task_type, content in task_templates.items():
            if task_type == "question_answering":
                for i, (question, answer) in enumerate(zip(content["questions"], content["answers"])):
                    example = {
                        "task_type": task_type,
                        "input": question,
                        "output": answer,
                        "generation": 0,
                        "condition": "human"
                    }
                    
                    if i < 6:
                        train_data.append(example)
                    elif i < 8:
                        val_data.append(example)
                    else:
                        test_data.append(example)
            
            elif task_type == "summarization":
                for i, (text, summary) in enumerate(zip(content["texts"], content["summaries"])):
                    example = {
                        "task_type": task_type,
                        "input": f"Summarize the following text: {text}",
                        "output": summary,
                        "generation": 0,
                        "condition": "human"
                    }
                    
                    if i < 2:
                        train_data.append(example)
                    elif i < 2:  # Small dataset for demonstration
                        val_data.append(example)
                    else:
                        test_data.append(example)
            
            elif task_type == "creative_writing":
                for i, (prompt, response) in enumerate(zip(content["prompts"], content["responses"])):
                    example = {
                        "task_type": task_type,
                        "input": prompt,
                        "output": response,
                        "generation": 0,
                        "condition": "human"
                    }
                    
                    if i < 3:
                        train_data.append(example)
                    elif i < 4:
                        val_data.append(example)
                    else:
                        test_data.append(example)
        
        # Create additional synthetic variations for larger dataset
        train_data_expanded = self._expand_dataset(train_data, self.config["dataset_config"]["train_size"])
        val_data_expanded = self._expand_dataset(val_data, self.config["dataset_config"]["val_size"])
        test_data_expanded = self._expand_dataset(test_data, self.config["dataset_config"]["test_size"])
        
        # Create Hugging Face datasets
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data_expanded),
            "validation": Dataset.from_list(val_data_expanded), 
            "test": Dataset.from_list(test_data_expanded)
        })
        
        # Save dataset
        dataset_path = self.data_dir / "generation_0_human_baseline.json"
        self._save_dataset(dataset_dict, dataset_path)
        
        logger.info(f"Generated base human dataset with {len(train_data_expanded)} training examples")
        return dataset_dict
    
    def _expand_dataset(self, base_data: List[Dict], target_size: int) -> List[Dict]:
        """Expand a small dataset to target size with variations."""
        if len(base_data) >= target_size:
            return random.sample(base_data, target_size)
        
        expanded_data = []
        while len(expanded_data) < target_size:
            # Sample from base data and add slight variations
            base_example = random.choice(base_data)
            
            # Create variation by adding small modifications
            expanded_example = base_example.copy()
            
            # Add slight variations to inputs and outputs
            if base_example["task_type"] == "question_answering":
                expanded_example["input"] = self._vary_question(base_example["input"])
                expanded_example["output"] = self._vary_answer(base_example["output"])
            elif base_example["task_type"] == "creative_writing":
                expanded_example["input"] = self._vary_creative_prompt(base_example["input"])
                expanded_example["output"] = self._vary_creative_response(base_example["output"])
            
            expanded_data.append(expanded_example)
        
        return expanded_data[:target_size]
    
    def _vary_question(self, question: str) -> str:
        """Add slight variation to a question."""
        variations = [
            question,
            f"Could you explain {question.lower().replace('what is', '').replace('?', '')}?",
            f"I'd like to know about {question.lower().replace('what is', '').replace('how do', '').replace('?', '')}.",
            f"Can you tell me {question.lower().replace('what is', 'what').replace('?', '')}?"
        ]
        return random.choice(variations)
    
    def _vary_answer(self, answer: str) -> str:
        """Add slight variation to an answer."""
        # Simple variations - in real implementation, would use more sophisticated methods
        variations = [
            answer,
            f"In simple terms, {answer.lower()}",
            f"Basically, {answer.lower()}",
            answer.replace("is", "can be described as")
        ]
        return random.choice(variations)
    
    def _vary_creative_prompt(self, prompt: str) -> str:
        """Add variation to creative writing prompt."""
        return prompt  # Keep prompts stable for this experiment
    
    def _vary_creative_response(self, response: str) -> str:
        """Add slight variation to creative response."""
        return response  # Keep responses relatively stable
    
    def generate_model_outputs(self, model, dataset: Dataset, generation: int, condition: str) -> List[Dict]:
        """Generate outputs from a trained model for the next generation's training data."""
        logger.info(f"Generating outputs for Generation {generation}, Condition: {condition}")
        
        # This would normally use the trained model to generate outputs
        # For demonstration purposes, we'll create synthetic "model outputs" that show degradation
        
        model_outputs = []
        for example in dataset:
            # Simulate model output with some degradation characteristics
            if generation > 1:
                output = self._simulate_degraded_output(example["output"], generation)
            else:
                output = self._simulate_model_output(example["output"])
            
            model_outputs.append({
                "task_type": example["task_type"],
                "input": example["input"], 
                "output": output,
                "generation": generation,
                "condition": condition,
                "parent_generation": generation - 1
            })
        
        return model_outputs
    
    def _simulate_model_output(self, original_output: str) -> str:
        """Simulate a model's output (with slight differences from human baseline)."""
        # Add slight model-like characteristics
        variations = [
            original_output,
            original_output.replace(".", ". Additionally,"),
            f"Based on my knowledge, {original_output.lower()}",
            original_output + " This is an important concept to understand."
        ]
        return random.choice(variations)
    
    def _simulate_degraded_output(self, original_output: str, generation: int) -> str:
        """Simulate degraded output from later generations."""
        # Simulate various forms of degradation
        degradation_factor = min(generation * 0.2, 0.8)
        
        if random.random() < degradation_factor:
            # Introduce various degradation patterns
            degradation_types = [
                self._add_repetition,
                self._reduce_specificity,
                self._add_generic_phrases,
                self._shorten_content
            ]
            
            degradation_func = random.choice(degradation_types)
            return degradation_func(original_output)
        
        return self._simulate_model_output(original_output)
    
    def _add_repetition(self, text: str) -> str:
        """Add repetitive elements (characteristic of model collapse)."""
        sentences = text.split('. ')
        if len(sentences) > 1:
            repeated_sentence = random.choice(sentences)
            return text + f" {repeated_sentence}"
        return text + " " + text.split()[-3:].__str__()
    
    def _reduce_specificity(self, text: str) -> str:
        """Make text more generic and less specific."""
        return text.replace("Paris", "the capital city").replace("France", "that country")
    
    def _add_generic_phrases(self, text: str) -> str:
        """Add generic, less informative phrases."""
        generic_phrases = [
            "It is important to note that",
            "Generally speaking,",
            "In many cases,",
            "This is a complex topic, but"
        ]
        return f"{random.choice(generic_phrases)} {text.lower()}"
    
    def _shorten_content(self, text: str) -> str:
        """Reduce content length (information loss)."""
        sentences = text.split('. ')
        if len(sentences) > 1:
            return '. '.join(sentences[:-1]) + '.'
        return text[:len(text)//2] + "..."
    
    def _save_dataset(self, dataset: DatasetDict, path: Path):
        """Save dataset to JSON file."""
        data_to_save = {}
        for split_name, split_data in dataset.items():
            data_to_save[split_name] = split_data.to_list()
        
        with open(path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        logger.info(f"Dataset saved to {path}")
    
    def load_dataset(self, path: Path) -> DatasetDict:
        """Load dataset from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        dataset_dict = DatasetDict({
            split_name: Dataset.from_list(split_data)
            for split_name, split_data in data.items()
        })
        
        return dataset_dict

if __name__ == "__main__":
    from config import CONFIG
    
    generator = MultiGenerationDataGenerator(CONFIG)
    
    # Generate base human dataset
    base_dataset = generator.generate_base_human_data()
    
    print(f"Base dataset created with {len(base_dataset['train'])} training examples")
    print("\nSample training example:")
    print(base_dataset['train'][0])