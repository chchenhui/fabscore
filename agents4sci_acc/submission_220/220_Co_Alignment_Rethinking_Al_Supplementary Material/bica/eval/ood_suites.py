"""
Out-of-Distribution (OOD) Evaluation Suites for BiCA
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import copy

from bica.envs import MapTalkEnv, OODMapTalkEnv


class OODEvaluator:
    """
    Evaluator for out-of-distribution performance
    
    Tests model robustness across different types of distribution shifts:
    1. Higher obstacle rates
    2. Sensor noise
    3. Structural patterns (corridors, rooms)
    4. Communication noise
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.ood_configs = self._create_ood_configs()
        
    def _create_ood_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create different OOD configuration variants"""
        configs = {}
        
        # Base OOD config
        base_ood = copy.deepcopy(self.base_config)
        base_ood['env']['ood'] = True
        
        # 1. High obstacle density
        configs['high_obstacles'] = copy.deepcopy(base_ood)
        configs['high_obstacles']['env']['ood_obstacle_rate'] = [0.4, 0.5]
        configs['high_obstacles']['env']['sensor_flip_prob'] = 0.0
        configs['high_obstacles']['env']['ood_patterns'] = ['random']
        
        # 2. Sensor noise
        configs['sensor_noise'] = copy.deepcopy(base_ood)
        configs['sensor_noise']['env']['ood_obstacle_rate'] = [0.25, 0.35]
        configs['sensor_noise']['env']['sensor_flip_prob'] = 0.15
        configs['sensor_noise']['env']['ood_patterns'] = ['random']
        
        # 3. Corridor patterns
        configs['corridors'] = copy.deepcopy(base_ood)
        configs['corridors']['env']['ood_obstacle_rate'] = [0.3, 0.4]
        configs['corridors']['env']['sensor_flip_prob'] = 0.05
        configs['corridors']['env']['ood_patterns'] = ['corridor']
        
        # 4. Room patterns
        configs['rooms'] = copy.deepcopy(base_ood)
        configs['rooms']['env']['ood_obstacle_rate'] = [0.3, 0.4]
        configs['rooms']['env']['sensor_flip_prob'] = 0.05
        configs['rooms']['env']['ood_patterns'] = ['rooms']
        
        # 5. Combined stress test
        configs['stress_test'] = copy.deepcopy(base_ood)
        configs['stress_test']['env']['ood_obstacle_rate'] = [0.45, 0.55]
        configs['stress_test']['env']['sensor_flip_prob'] = 0.2
        configs['stress_test']['env']['ood_patterns'] = ['corridor', 'rooms']
        
        # 6. Communication noise
        configs['comm_noise'] = copy.deepcopy(base_ood)
        configs['comm_noise']['env']['ood_obstacle_rate'] = [0.25, 0.35]
        configs['comm_noise']['env']['sensor_flip_prob'] = 0.05
        configs['comm_noise']['env']['ood_patterns'] = ['random']
        configs['comm_noise']['env']['communication_noise'] = 0.15  # Higher protocol noise
        
        return configs
    
    def evaluate_ood_suite(self, 
                          models: Dict[str, torch.nn.Module],
                          num_episodes: int = 50) -> Dict[str, Dict[str, float]]:
        """
        Evaluate models on full OOD suite
        
        Args:
            models: Dictionary of trained models
            num_episodes: Number of episodes per OOD variant
            
        Returns:
            results: Dictionary of results per OOD variant
        """
        results = {}
        
        for ood_name, ood_config in self.ood_configs.items():
            print(f"Evaluating OOD variant: {ood_name}")
            
            # Create OOD environment
            env = self._create_ood_env(ood_config)
            
            # Run evaluation
            variant_results = self._evaluate_on_env(env, models, num_episodes)
            results[ood_name] = variant_results
            
            print(f"  Success rate: {variant_results['success_rate']:.3f}")
            print(f"  Collision rate: {variant_results['collision_rate']:.3f}")
            print(f"  Avg steps: {variant_results['avg_steps']:.1f}")
        
        # Compute aggregate OOD metrics
        results['aggregate'] = self._compute_aggregate_metrics(results)
        
        return results
    
    def _create_ood_env(self, config: Dict[str, Any]) -> OODMapTalkEnv:
        """Create OOD environment from config"""
        from bica.envs import create_env
        return create_env(config['env'])
    
    def _evaluate_on_env(self, 
                        env: OODMapTalkEnv,
                        models: Dict[str, torch.nn.Module],
                        num_episodes: int) -> Dict[str, float]:
        """
        Evaluate models on a specific OOD environment
        
        Args:
            env: OOD environment
            models: Dictionary of models
            num_episodes: Number of episodes
            
        Returns:
            metrics: Evaluation metrics
        """
        # Extract models
        ai_policy = models['ai_policy']
        human_surrogate = models['human_surrogate']
        protocol_generator = models.get('protocol_generator')
        instructor = models.get('instructor')
        
        # Evaluation metrics
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        collision_count = 0
        timeout_count = 0
        
        # Communication metrics
        message_counts = []
        protocol_diversity = []
        
        # Confidence tracking for calibration
        confidences = []
        accuracies = []
        
        device = next(ai_policy.parameters()).device
        
        for episode in range(num_episodes):
            # Reset environment
            obs = env.reset()
            
            # Reset model states
            ai_hidden = None
            human_hidden = human_surrogate.init_hidden(1, device) if human_surrogate else None
            instructor_hidden = None
            
            episode_reward = 0.0
            episode_messages = 0
            episode_protocols = []
            
            # Episode loop
            for step in range(env.max_steps):
                # Preprocess observations
                ai_obs_tensor = self._preprocess_ai_obs(obs).unsqueeze(0).to(device)
                human_obs_tensor = self._preprocess_human_obs(obs).unsqueeze(0).to(device)
                
                # Generate protocol message (if available)
                ai_message_idx = 0
                if protocol_generator is not None:
                    context = self._build_context(env, step).unsqueeze(0).to(device)
                    ai_message, _ = protocol_generator.sample_message(context, tau=0.5)
                    ai_message_idx = ai_message.item()
                    episode_protocols.append(ai_message_idx)
                
                # Instructor intervention (if available)
                instructor_action_idx = 0
                if instructor is not None:
                    history_features = self._build_history_features(env, step).unsqueeze(0).unsqueeze(0).to(device)
                    instructor_action, _, instructor_hidden = instructor.sample_intervention(
                        history_features, instructor_hidden
                    )
                    instructor_action_idx = instructor_action.item()
                
                # Human message
                human_message_idx = 0
                if human_surrogate is not None:
                    ai_msg_tensor = torch.tensor([ai_message_idx]).to(device)
                    instr_tensor = torch.tensor([instructor_action_idx]).to(device)
                    human_message, _, human_hidden = human_surrogate.sample_message(
                        human_obs_tensor, ai_msg_tensor, instr_tensor, human_hidden
                    )
                    human_message_idx = human_message.item()
                    episode_messages += 1
                
                # AI action
                human_msg_tensor = torch.tensor([human_message_idx]).to(device)
                ai_action, ai_log_prob, ai_hidden = ai_policy.sample_action(
                    ai_obs_tensor, human_msg_tensor, ai_hidden
                )
                ai_action_idx = ai_action.item()
                
                # Track confidence (action probability)
                with torch.no_grad():
                    ai_probs, _ = ai_policy.get_action_probs(ai_obs_tensor, human_msg_tensor, ai_hidden)
                    confidence = ai_probs.max().item()
                    confidences.append(confidence)
                
                # Environment step
                next_obs, reward, done, info = env.step(
                    ai_action_idx, ai_message_idx, human_message_idx, instructor_action_idx
                )
                
                episode_reward += reward
                
                # Track accuracy (whether action led to positive reward)
                accuracies.append(1.0 if reward > -1.0 else 0.0)  # Not collision/timeout
                
                if done:
                    if info.get('success', False):
                        success_count += 1
                    if info.get('collision', False):
                        collision_count += 1
                    if step >= env.max_steps - 1:
                        timeout_count += 1
                    break
                
                obs = next_obs
            
            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            message_counts.append(episode_messages)
            
            # Protocol diversity (unique protocols used)
            if episode_protocols:
                protocol_diversity.append(len(set(episode_protocols)) / len(episode_protocols))
            else:
                protocol_diversity.append(0.0)
        
        # Compute metrics
        metrics = {
            'success_rate': success_count / num_episodes,
            'collision_rate': collision_count / num_episodes,
            'timeout_rate': timeout_count / num_episodes,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_steps': np.mean(episode_lengths),
            'std_steps': np.std(episode_lengths),
            'avg_messages': np.mean(message_counts),
            'protocol_diversity': np.mean(protocol_diversity),
            'miscalibration': self._compute_calibration_error(confidences, accuracies)
        }
        
        return metrics
    
    def _preprocess_ai_obs(self, obs: Dict) -> torch.Tensor:
        """Preprocess AI observation"""
        from bica.models.policy import preprocess_ai_observation
        return preprocess_ai_observation(obs['ai_obs'], obs['ai_heading'])
    
    def _preprocess_human_obs(self, obs: Dict) -> torch.Tensor:
        """Preprocess human observation"""
        from bica.models.human_surrogate import preprocess_human_observation
        return preprocess_human_observation(obs['human_obs'])
    
    def _build_context(self, env: OODMapTalkEnv, step: int) -> torch.Tensor:
        """Build context vector for protocol generator"""
        from bica.models.protocol import ContextBuilder
        
        builder = ContextBuilder()
        env_state = {
            'agent_pos': env.agent_pos,
            'goal_pos': env.goal_pos,
            'step_count': step,
            'distance_to_goal': np.linalg.norm(env.agent_pos - env.goal_pos),
            'obstacle_density': np.mean(env.grid)
        }
        history = {'message_history': list(env.message_history)}
        
        return builder.build_context(env_state, history)
    
    def _build_history_features(self, env: OODMapTalkEnv, step: int) -> torch.Tensor:
        """Build history features for instructor"""
        from bica.models.instructor import HistoryFeatureExtractor
        
        extractor = HistoryFeatureExtractor()
        env_state = {
            'step_count': step,
            'distance_to_goal': np.linalg.norm(env.agent_pos - env.goal_pos),
            'model_confidence': 0.5,  # Placeholder
            'ood_detected': True
        }
        
        features = extractor.extract_features(env_state)
        return torch.from_numpy(features).float()
    
    def _compute_calibration_error(self, 
                                  confidences: List[float], 
                                  accuracies: List[float],
                                  num_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)"""
        if not confidences or not accuracies:
            return 0.0
        
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = accuracies[in_bin].mean()
                # Average confidence in this bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                # ECE contribution
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _compute_aggregate_metrics(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute aggregate metrics across all OOD variants"""
        # Exclude 'aggregate' key if it exists
        variant_results = {k: v for k, v in results.items() if k != 'aggregate'}
        
        if not variant_results:
            return {}
        
        # Compute means across variants
        aggregate = {}
        metric_names = list(next(iter(variant_results.values())).keys())
        
        for metric_name in metric_names:
            values = [variant[metric_name] for variant in variant_results.values()]
            aggregate[f'mean_{metric_name}'] = np.mean(values)
            aggregate[f'std_{metric_name}'] = np.std(values)
            aggregate[f'min_{metric_name}'] = np.min(values)
            aggregate[f'max_{metric_name}'] = np.max(values)
        
        # Compute robustness score (higher is better)
        success_rates = [variant['success_rate'] for variant in variant_results.values()]
        collision_rates = [variant['collision_rate'] for variant in variant_results.values()]
        
        # Robustness = average success - average collision - std(success)
        robustness_score = (np.mean(success_rates) - 
                           np.mean(collision_rates) - 
                           np.std(success_rates))
        
        aggregate['robustness_score'] = max(0.0, robustness_score)
        
        return aggregate
    
    def compare_with_baseline(self, 
                             results: Dict[str, Dict[str, float]],
                             baseline_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Compare OOD results with baseline
        
        Args:
            results: Current model OOD results
            baseline_results: Baseline model OOD results
            
        Returns:
            comparison: Comparison metrics
        """
        comparison = {}
        
        for variant_name in results.keys():
            if variant_name in baseline_results:
                variant_comparison = {}
                
                current = results[variant_name]
                baseline = baseline_results[variant_name]
                
                # Compute improvements
                for metric_name in current.keys():
                    if metric_name in baseline:
                        current_val = current[metric_name]
                        baseline_val = baseline[metric_name]
                        
                        if baseline_val != 0:
                            improvement = (current_val - baseline_val) / abs(baseline_val)
                        else:
                            improvement = 0.0
                        
                        variant_comparison[f'{metric_name}_improvement'] = improvement
                        variant_comparison[f'{metric_name}_current'] = current_val
                        variant_comparison[f'{metric_name}_baseline'] = baseline_val
                
                comparison[variant_name] = variant_comparison
        
        return comparison
    
    def generate_ood_report(self, 
                           results: Dict[str, Dict[str, float]],
                           baseline_results: Optional[Dict[str, Dict[str, float]]] = None) -> str:
        """
        Generate a human-readable OOD evaluation report
        
        Args:
            results: OOD evaluation results
            baseline_results: Optional baseline results for comparison
            
        Returns:
            report: Formatted report string
        """
        report = ["=" * 60]
        report.append("OOD EVALUATION REPORT")
        report.append("=" * 60)
        
        # Aggregate results first
        if 'aggregate' in results:
            agg = results['aggregate']
            report.append("\nAGGREGATE METRICS:")
            report.append("-" * 30)
            report.append(f"Mean Success Rate: {agg.get('mean_success_rate', 0):.3f} ± {agg.get('std_success_rate', 0):.3f}")
            report.append(f"Mean Collision Rate: {agg.get('mean_collision_rate', 0):.3f} ± {agg.get('std_collision_rate', 0):.3f}")
            report.append(f"Robustness Score: {agg.get('robustness_score', 0):.3f}")
            report.append(f"Mean Miscalibration: {agg.get('mean_miscalibration', 0):.3f}")
        
        # Individual variant results
        for variant_name, metrics in results.items():
            if variant_name == 'aggregate':
                continue
                
            report.append(f"\n{variant_name.upper()}:")
            report.append("-" * 30)
            report.append(f"Success Rate: {metrics['success_rate']:.3f}")
            report.append(f"Collision Rate: {metrics['collision_rate']:.3f}")
            report.append(f"Avg Steps: {metrics['avg_steps']:.1f}")
            report.append(f"Protocol Diversity: {metrics['protocol_diversity']:.3f}")
            report.append(f"Miscalibration (ECE): {metrics['miscalibration']:.3f}")
            
            # Comparison with baseline if available
            if baseline_results and variant_name in baseline_results:
                baseline = baseline_results[variant_name]
                success_improvement = ((metrics['success_rate'] - baseline['success_rate']) / 
                                     max(baseline['success_rate'], 0.001)) * 100
                collision_improvement = ((baseline['collision_rate'] - metrics['collision_rate']) / 
                                       max(baseline['collision_rate'], 0.001)) * 100
                
                report.append(f"Success Improvement: {success_improvement:+.1f}%")
                report.append(f"Collision Reduction: {collision_improvement:+.1f}%")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def create_ood_evaluator(config: Dict[str, Any]) -> OODEvaluator:
    """Factory function to create OOD evaluator"""
    return OODEvaluator(config)
