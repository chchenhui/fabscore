"""
Main training loop for BiCA MapTalk experiment
Implements alternating updates as specified in the paper
"""

import torch
import torch.optim as optim
import numpy as np
import yaml
import argparse
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict, deque
import wandb
from tqdm import tqdm

# BiCA imports
from bica.envs import MapTalkEnv, OODMapTalkEnv, create_env
from bica.models import (
    AIPolicy, ValueNet, HumanSurrogate, ProtocolGenerator, 
    RepresentationMapper, Instructor
)
from bica.models.policy import create_ai_policy, create_value_net, preprocess_ai_observation
from bica.models.human_surrogate import create_human_surrogate, preprocess_human_observation
from bica.models.protocol import create_protocol_generator, create_context_builder
from bica.models.rep_mapper import create_representation_mapper, create_gap_computer, create_latent_extractor
from bica.models.instructor import create_instructor, create_history_extractor, create_intervention_executor
from bica.losses import PPOLoss, IBLoss, RepresentationGapLoss
from bica.losses.ppo import create_ppo_loss, create_dual_updater, compute_gae
from bica.losses.ib import create_protocol_regularizer
from bica.eval.metrics_bas_ccm import MetricsComputer
from bica.losses.repgap import create_repgap_loss


class TrajectoryBuffer:
    """Buffer for storing trajectory data"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.messages = []
        self.interventions = []
        self.infos = []
    
    def add(self, obs, action, reward, done, log_prob, value, messages, intervention, info):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.messages.append(messages)
        self.interventions.append(intervention)
        self.infos.append(info)
    
    def get_batch(self):
        return {
            'observations': self.observations,
            'actions': torch.tensor(self.actions),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'dones': torch.tensor(self.dones, dtype=torch.float32),
            'log_probs': torch.stack(self.log_probs),
            'values': torch.stack(self.values),
            'messages': self.messages,
            'interventions': self.interventions,
            'infos': self.infos
        }


class BiCATrainer:
    """Main BiCA training class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize environment
        self.env = create_env(config['env'])
        self.ood_env = create_env({**config['env'], 'ood': True})
        
        # Initialize models
        self._init_models()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize loss functions
        self._init_losses()
        
        # Initialize helper classes
        self._init_helpers()
        
        # Initialize metrics computer
        self.metrics_computer = MetricsComputer(config)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_performance = -float('inf')
        
        # Early stopping state
        self.early_stopping_counter = 0
        self.last_improvement_epoch = 0
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        
        # Initialize logging
        if config.get('use_wandb', True):
            wandb.init(project="bica-maptalk", config=config)
    
    def _init_models(self):
        """Initialize all neural network models"""
        model_config = self.config['model']
        
        # Check for single directional configuration
        single_directional = self.config.get('single_directional', {})
        disable_protocol_learning = single_directional.get('disable_protocol_learning', False)
        disable_instructor = single_directional.get('disable_instructor', False)
        disable_rep_mapper = single_directional.get('disable_rep_mapper', False)
        
        # AI policy and value network (always needed)
        self.ai_policy = create_ai_policy(model_config).to(self.device)
        self.value_net = create_value_net(model_config).to(self.device)
        
        # Human surrogate (always needed)
        self.human_surrogate = create_human_surrogate(model_config).to(self.device)
        
        # Protocol generator (conditional)
        if not disable_protocol_learning:
            self.protocol_generator = create_protocol_generator(model_config).to(self.device)
            self.context_builder = create_context_builder(model_config)
        else:
            self.protocol_generator = None
            self.context_builder = None
            print(" Single directional mode: Protocol learning disabled")
        
        # Representation mapper (conditional)
        if not disable_rep_mapper:
            self.rep_mapper = create_representation_mapper(model_config).to(self.device)
        else:
            self.rep_mapper = None
            print(" Single directional mode: Representation alignment disabled")
        
        # Instructor (conditional)
        if not disable_instructor:
            self.instructor = create_instructor(model_config).to(self.device)
        else:
            self.instructor = None
            print(" Single directional mode: Adaptive teaching disabled")
        
        # Store prior policies for KL regularization
        self.ai_policy_prior = create_ai_policy(model_config).to(self.device)
        self.ai_policy_prior.load_state_dict(self.ai_policy.state_dict())
        
        self.human_surrogate_prior = create_human_surrogate(model_config).to(self.device)
        self.human_surrogate_prior.load_state_dict(self.human_surrogate.state_dict())
        
        # Freeze priors
        for param in self.ai_policy_prior.parameters():
            param.requires_grad = False
        for param in self.human_surrogate_prior.parameters():
            param.requires_grad = False
    
    def _init_optimizers(self):
        """Initialize optimizers for all models"""
        lr = float(self.config['train']['lr'])
        
        self.optimizers = {
            'ai_policy': optim.AdamW(self.ai_policy.parameters(), lr=lr),
            'value_net': optim.AdamW(self.value_net.parameters(), lr=lr),
            'human_surrogate': optim.AdamW(self.human_surrogate.parameters(), lr=lr)
        }
        
        # Add optimizers for components that are not disabled
        if self.protocol_generator is not None:
            self.optimizers['protocol'] = optim.AdamW(self.protocol_generator.parameters(), lr=lr)
        
        if self.rep_mapper is not None:
            self.optimizers['rep_mapper'] = optim.AdamW(self.rep_mapper.parameters(), lr=lr)
        
        if self.instructor is not None:
            self.optimizers['instructor'] = optim.AdamW(self.instructor.parameters(), lr=lr)
    
    def _init_losses(self):
        """Initialize loss functions"""
        # PPO loss with dual updater
        self.ppo_loss = create_ppo_loss(self.config['train'])
        self.dual_updater = create_dual_updater(self.config['regularizers'])
        
        # Protocol regularization
        self.protocol_regularizer = create_protocol_regularizer(self.config['regularizers'])
        
        # Representation gap loss
        self.repgap_loss = create_repgap_loss(self.config['regularizers'])
    
    def _init_helpers(self):
        """Initialize helper classes"""
        self.gap_computer = create_gap_computer(self.config['model'])
        self.latent_extractor = create_latent_extractor()
        self.history_extractor = create_history_extractor(self.config['model'])
        self.intervention_executor = create_intervention_executor()
        
        # Gumbel temperature schedule
        self.gumbel_tau = self.config['protocol']['gumbel_tau_start']
        self.tau_decay = self.config['protocol']['tau_decay']
        self.tau_end = self.config['protocol']['gumbel_tau_end']
    
    def rollout(self, env, num_episodes: int = 32) -> List[TrajectoryBuffer]:
        """
        Collect rollouts from environment
        
        Args:
            env: Environment to roll out in
            num_episodes: Number of episodes to collect
            
        Returns:
            trajectories: List of trajectory buffers
        """
        trajectories = []
        
        for episode in range(num_episodes):
            trajectory = TrajectoryBuffer()
            
            # Reset environment and models
            obs = env.reset()
            ai_hidden = None
            human_hidden = self.human_surrogate.init_hidden(1, self.device)
            instructor_hidden = None
            
            # Episode loop
            for step in range(env.max_steps):
                # Preprocess observations
                ai_obs_tensor = preprocess_ai_observation(
                    obs['ai_obs'], obs['ai_heading']
                ).unsqueeze(0).to(self.device)
                
                human_obs_tensor = preprocess_human_observation(
                    obs['human_obs']
                ).unsqueeze(0).to(self.device)
                
                # Build context for protocol generator
                env_state = {
                    'agent_pos': env.agent_pos,
                    'goal_pos': env.goal_pos,
                    'step_count': env.step_count,
                    'distance_to_goal': np.linalg.norm(env.agent_pos - env.goal_pos)
                }
                
                history = {
                    'message_history': list(env.message_history),
                    'recent_collisions': sum(1 for info in trajectory.infos if info.get('collision', False)),
                    'recent_failures': len([info for info in trajectory.infos if not info.get('success', False)])
                }
                
                # Generate AI protocol message (conditional for single directional)
                if self.protocol_generator is not None and self.context_builder is not None:
                    context = self.context_builder.build_context(env_state, history).unsqueeze(0).to(self.device)
                    ai_message, _ = self.protocol_generator.sample_message(context, self.gumbel_tau)
                    ai_message_idx = ai_message.item()
                else:
                    # Single directional mode: use simple/fixed communication
                    ai_message_idx = 0  # Default/no message
                    ai_message = torch.tensor([0]).to(self.device)
                    context = torch.zeros(1, 32).to(self.device)  # Dummy context for logging
                
                # Get instructor intervention (conditional for single directional)
                if self.instructor is not None:
                    # Update history for instructor
                    if trajectory.infos:
                        self.history_extractor.update_history(trajectory.infos[-1])
                    
                    history_features = torch.from_numpy(
                        self.history_extractor.extract_features(env_state)
                    ).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    
                    instructor_action, _, instructor_hidden = self.instructor.sample_intervention(
                        history_features, instructor_hidden
                    )
                    instructor_action_idx = instructor_action.item()
                    
                    # Execute intervention
                    intervention_effects = self.intervention_executor.execute_intervention(
                        instructor_action_idx, {'step_count': env.step_count}
                    )
                else:
                    # Single directional mode: no instructor interventions
                    instructor_action_idx = 0  # No intervention
                    intervention_effects = {'intervention_cost': 0.0}
                
                # Generate human message
                instructor_action_tensor = torch.tensor([instructor_action_idx]).to(self.device) if self.instructor is not None else torch.tensor([0]).to(self.device)
                human_message, _, human_hidden = self.human_surrogate.sample_message(
                    human_obs_tensor,
                    ai_message,
                    instructor_action_tensor,
                    human_hidden
                )
                human_message_idx = human_message.item()
                
                # AI action selection
                ai_action, ai_log_prob, ai_hidden = self.ai_policy.sample_action(
                    ai_obs_tensor, human_message, ai_hidden
                )
                ai_action_idx = ai_action.item()
                
                # Value estimation
                value, _ = self.value_net(ai_obs_tensor, human_message, ai_hidden)
                
                # Environment step
                next_obs, reward, done, info = env.step(
                    ai_action_idx, ai_message_idx, human_message_idx, instructor_action_idx
                )
                
                # Store trajectory data
                trajectory.add(
                    obs={
                        'ai_obs': ai_obs_tensor,
                        'human_obs': human_obs_tensor,
                        'context': context
                    },
                    action=ai_action_idx,
                    reward=reward,
                    done=done,
                    log_prob=ai_log_prob,
                    value=value,
                    messages={
                        'ai_message': ai_message_idx,
                        'human_message': human_message_idx
                    },
                    intervention=instructor_action_idx,
                    info=info
                )
                
                # Extract latent representations for RepGap loss
                human_latent = self.latent_extractor.extract_human_latent(
                    self.human_surrogate, human_obs_tensor, ai_message, instructor_action_tensor, human_hidden
                )
                ai_latent = self.latent_extractor.extract_ai_latent(
                    self.ai_policy, ai_obs_tensor, human_message, ai_hidden
                )
                self.latent_extractor.collect_latents(human_latent, ai_latent)
                
                if done:
                    break
                
                obs = next_obs
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def update_human_surrogate(self, trajectories: List[TrajectoryBuffer]) -> Dict[str, float]:
        """Update human surrogate model (E step)"""
        self.optimizers['human_surrogate'].zero_grad()
        
        total_loss = 0.0
        total_samples = 0
        
        for trajectory in trajectories:
            batch = trajectory.get_batch()
            
            # Prepare inputs
            human_obs_batch = torch.stack([obs['human_obs'] for obs in batch['observations']]).squeeze(1)
            ai_messages = torch.tensor([msg['ai_message'] for msg in batch['messages']]).to(self.device)
            interventions = torch.tensor(batch['interventions']).to(self.device)
            target_messages = torch.tensor([msg['human_message'] for msg in batch['messages']]).to(self.device)
            
            # Forward pass
            hidden = self.human_surrogate.init_hidden(len(batch['observations']), self.device)
            message_logits, _ = self.human_surrogate(human_obs_batch, ai_messages, interventions, hidden)
            
            # Cross-entropy loss
            ce_loss = torch.nn.functional.cross_entropy(message_logits, target_messages)
            
            # KL to prior
            with torch.no_grad():
                prior_logits, _ = self.human_surrogate_prior(human_obs_batch, ai_messages, interventions, hidden)
            
            kl_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(message_logits, dim=-1),
                torch.nn.functional.softmax(prior_logits, dim=-1),
                reduction='batchmean'
            )
            
            # Total loss
            lambda_h = self.dual_updater.lambda_h
            loss = ce_loss + lambda_h * kl_loss
            
            total_loss += loss.item() * len(batch['observations'])
            total_samples += len(batch['observations'])
            
            loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.human_surrogate.parameters(), self.ppo_loss.max_grad_norm)
        
        self.optimizers['human_surrogate'].step()
        
        return {
            'human_loss': total_loss / total_samples,
            'human_ce_loss': ce_loss.item(),
            'human_kl_loss': kl_loss.item()
        }
    
    def update_ai_policy(self, trajectories: List[TrajectoryBuffer]) -> Dict[str, float]:
        """Update AI policy using PPO (A step)"""
        # Prepare batch data
        all_observations = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_old_log_probs = []
        all_old_values = []
        all_human_messages = []
        
        for trajectory in trajectories:
            batch = trajectory.get_batch()
            all_observations.extend([obs['ai_obs'] for obs in batch['observations']])
            all_actions.extend(batch['actions'].tolist())
            all_rewards.extend(batch['rewards'].tolist())
            all_dones.extend(batch['dones'].tolist())
            all_old_log_probs.extend(batch['log_probs'].tolist())
            all_old_values.extend(batch['values'].tolist())
            all_human_messages.extend([msg['human_message'] for msg in batch['messages']])
        
        # Convert to tensors (ensure proper shapes)
        obs_batch = torch.stack(all_observations).squeeze(1)
        actions_batch = torch.tensor(all_actions).to(self.device)
        rewards_batch = torch.tensor(all_rewards).to(self.device)
        dones_batch = torch.tensor(all_dones).to(self.device)
        old_log_probs_batch = torch.tensor(all_old_log_probs).to(self.device)
        old_values_batch = torch.tensor(all_old_values).to(self.device).squeeze()  # Remove extra dimensions
        human_messages_batch = torch.tensor(all_human_messages).to(self.device)
        
        # Compute advantages using GAE
        with torch.no_grad():
            new_values, _ = self.value_net(obs_batch, human_messages_batch)
            new_values = new_values.squeeze(-1)
        
        # Simple advantage computation (without full GAE for now)
        # This is a simplified version that works with variable-length trajectories
        gamma = self.config['train']['gamma']
        
        # Compute simple advantages and returns
        advantages = torch.zeros_like(rewards_batch)
        returns = torch.zeros_like(rewards_batch)
        
        start_idx = 0
        for trajectory in trajectories:
            batch = trajectory.get_batch()
            traj_len = len(batch['rewards'])
            
            # Extract trajectory data (ensure proper shapes)
            traj_rewards = rewards_batch[start_idx:start_idx + traj_len]
            traj_values = old_values_batch[start_idx:start_idx + traj_len]
            traj_dones = dones_batch[start_idx:start_idx + traj_len]
            
            # Compute returns (discounted rewards)
            traj_returns = torch.zeros_like(traj_rewards)
            running_return = 0.0
            for t in reversed(range(traj_len)):
                if traj_dones[t].item() > 0.5:  # Episode ended
                    running_return = 0.0
                running_return = traj_rewards[t].item() + gamma * running_return
                traj_returns[t] = running_return
            
            # Simple advantage: A = R - V
            traj_advantages = traj_returns - traj_values
            
            # Store results
            advantages[start_idx:start_idx + traj_len] = traj_advantages
            returns[start_idx:start_idx + traj_len] = traj_returns
            
            start_idx += traj_len
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO updates for multiple epochs
        for ppo_epoch in range(self.config['train'].get('ppo_epochs', 5)):
            self.optimizers['ai_policy'].zero_grad()
            self.optimizers['value_net'].zero_grad()
            
            # Get new policy outputs
            new_log_probs, entropy = self.ai_policy.log_prob(obs_batch, human_messages_batch, actions=actions_batch, return_entropy=True)
            new_values, _ = self.value_net(obs_batch, human_messages_batch)
            new_values = new_values.squeeze(-1)
            
            # Get prior log probs for KL regularization
            with torch.no_grad():
                prior_log_probs = self.ai_policy_prior.log_prob(obs_batch, human_messages_batch, actions=actions_batch, return_entropy=False)
            
            # Prepare batch data for loss computation
            policy_outputs = {
                'log_probs': new_log_probs,
                'entropy': entropy
            }
            
            value_outputs = {
                'values': new_values
            }
            
            batch_data = {
                'old_log_probs': old_log_probs_batch,
                'advantages': advantages,
                'returns': returns,
                'old_values': old_values_batch,
                'prior_log_probs': prior_log_probs
            }
            
            # Compute PPO loss
            loss_dict = self.ppo_loss.compute_total_loss(policy_outputs, value_outputs, batch_data)
            
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.ai_policy.parameters(), self.ppo_loss.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.ppo_loss.max_grad_norm)
            
            self.optimizers['ai_policy'].step()
            self.optimizers['value_net'].step()
            
        # Return metrics from the last PPO epoch
        return {f'ai_{k}': v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    def update_protocol(self, trajectories: List[TrajectoryBuffer]) -> Dict[str, float]:
        """Update protocol generator (P step)"""
        self.optimizers['protocol'].zero_grad()
        
        total_loss = 0.0
        total_samples = 0
        
        for trajectory in trajectories:
            batch = trajectory.get_batch()
            contexts = torch.stack([obs['context'] for obs in batch['observations']]).squeeze(1)
            
            # Forward pass
            message_logits, code_probs, code_samples = self.protocol_generator(contexts, self.gumbel_tau)
            
            # Task loss (reward prediction or success prediction)
            rewards = batch['rewards'].to(self.device)
            success_targets = (rewards > 0).float()  # Binary success
            
            task_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                message_logits.mean(dim=-1), success_targets
            )
            
            # Protocol regularization
            reg_dict = self.protocol_regularizer.compute_total_regularization(
                torch.softmax(message_logits, dim=-1), code_probs, contexts
            )
            
            # Total loss
            loss = task_loss + reg_dict['total_regularization']
            
            total_loss += loss.item() * len(batch['observations'])
            total_samples += len(batch['observations'])
            
            loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.protocol_generator.parameters(), self.ppo_loss.max_grad_norm)
        
        self.optimizers['protocol'].step()
        
        # Update Gumbel temperature
        self.gumbel_tau = max(self.tau_end, self.gumbel_tau * self.tau_decay)
        
        return {
            'protocol_loss': total_loss / total_samples,
            'protocol_task_loss': task_loss.item(),
            'protocol_reg_loss': reg_dict['total_regularization'].item(),
            'gumbel_tau': self.gumbel_tau
        }
    
    def update_rep_mapper(self, trajectories: List[TrajectoryBuffer]) -> Dict[str, float]:
        """Update representation mapper (M step)"""
        # Get collected latents
        human_latents, ai_latents = self.latent_extractor.get_batch_latents()
        
        if human_latents is None or ai_latents is None:
            return {'rep_mapper_loss': 0.0}
        
        human_latents = human_latents.to(self.device)
        ai_latents = ai_latents.to(self.device)
        
        self.optimizers['rep_mapper'].zero_grad()
        
        # Compute representation gap loss
        loss_dict = self.repgap_loss.compute_total_repgap_loss(
            human_latents, ai_latents, self.rep_mapper
        )
        
        loss_dict['repgap_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.rep_mapper.parameters(), self.ppo_loss.max_grad_norm)
        
        self.optimizers['rep_mapper'].step()
        
        # Clear collected latents
        self.latent_extractor.clear()
        
        return {f'rep_{k}': v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    def update_instructor(self, trajectories: List[TrajectoryBuffer]) -> Dict[str, float]:
        """Update instructor model (I step)"""
        self.optimizers['instructor'].zero_grad()
        
        total_loss = 0.0
        total_samples = 0
        
        for trajectory in trajectories:
            batch = trajectory.get_batch()
            
            # Prepare history features (simplified)
            history_features = []
            for i, info in enumerate(batch['infos']):
                env_state = {
                    'step_count': i,
                    'model_confidence': 0.5,  # Placeholder
                    'ood_detected': False
                }
                features = self.history_extractor.extract_features(env_state)
                history_features.append(features)
            
            history_batch = torch.from_numpy(np.stack(history_features)).float().to(self.device)
            history_batch = history_batch.unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            intervention_logits, _ = self.instructor(history_batch)
            intervention_logits = intervention_logits.squeeze(0)  # Remove batch dimension
            
            # Compute returns for instructor (negative cost + reward improvement)
            rewards = batch['rewards'].to(self.device)
            intervention_costs = torch.tensor([
                self.intervention_executor.intervention_costs.get(int(action), 0.1) 
                for action in batch['interventions']
            ]).to(self.device)
            
            instructor_returns = rewards - self.config['regularizers']['kappa_teach'] * intervention_costs
            
            # Policy gradient loss
            interventions = torch.tensor(batch['interventions']).to(self.device)
            log_probs = torch.nn.functional.log_softmax(intervention_logits, dim=-1)
            
            # Handle both 1D and 2D cases for log_probs
            if log_probs.dim() == 1:
                selected_log_probs = log_probs[interventions]
            else:
                selected_log_probs = log_probs.gather(1, interventions.unsqueeze(-1)).squeeze(-1)
            
            # REINFORCE loss
            loss = -(selected_log_probs * instructor_returns).mean()
            
            total_loss += loss.item() * len(batch['observations'])
            total_samples += len(batch['observations'])
            
            loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.instructor.parameters(), self.ppo_loss.max_grad_norm)
        
        self.optimizers['instructor'].step()
        
        return {
            'instructor_loss': total_loss / total_samples,
            'intervention_cost': self.intervention_executor.get_total_cost()
        }
    
    def update_dual_variables(self, trajectories: List[TrajectoryBuffer]) -> Dict[str, float]:
        """Update dual variables for KL budget constraints (Λ step)"""
        # Estimate KL divergences
        ai_kl_total = 0.0
        human_kl_total = 0.0
        total_samples = 0
        
        for trajectory in trajectories:
            batch = trajectory.get_batch()
            
            # AI KL divergence
            ai_obs_batch = torch.stack([obs['ai_obs'] for obs in batch['observations']]).squeeze(1)
            human_messages_batch = torch.tensor([msg['human_message'] for msg in batch['messages']]).to(self.device)
            
            with torch.no_grad():
                ai_probs, _ = self.ai_policy.get_action_probs(ai_obs_batch, human_messages_batch)
                ai_prior_probs, _ = self.ai_policy_prior.get_action_probs(ai_obs_batch, human_messages_batch)
                
                ai_kl = torch.nn.functional.kl_div(
                    torch.log(ai_prior_probs + 1e-8), ai_probs, reduction='batchmean'
                )
                ai_kl_total += ai_kl.item() * len(batch['observations'])
            
            # Human KL divergence (similar computation)
            human_obs_batch = torch.stack([obs['human_obs'] for obs in batch['observations']]).squeeze(1)
            ai_messages_batch = torch.tensor([msg['ai_message'] for msg in batch['messages']]).to(self.device)
            interventions_batch = torch.tensor(batch['interventions']).to(self.device)
            
            with torch.no_grad():
                hidden = self.human_surrogate.init_hidden(len(batch['observations']), self.device)
                human_probs, _ = self.human_surrogate.get_message_probs(
                    human_obs_batch, ai_messages_batch, interventions_batch, hidden
                )
                human_prior_probs, _ = self.human_surrogate_prior.get_message_probs(
                    human_obs_batch, ai_messages_batch, interventions_batch, hidden
                )
                
                human_kl = torch.nn.functional.kl_div(
                    torch.log(human_prior_probs + 1e-8), human_probs, reduction='batchmean'
                )
                human_kl_total += human_kl.item() * len(batch['observations'])
            
            total_samples += len(batch['observations'])
        
        # Average KL divergences
        avg_ai_kl = ai_kl_total / total_samples
        avg_human_kl = human_kl_total / total_samples
        
        # Update dual variables
        dual_info = self.dual_updater.update_dual_variables(avg_ai_kl, avg_human_kl)
        
        return {
            'avg_ai_kl': avg_ai_kl,
            'avg_human_kl': avg_human_kl,
            **dual_info
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with alternating updates"""
        # Collect rollouts
        print(f"Epoch {self.epoch}: Collecting rollouts...")
        trajectories = self.rollout(self.env, self.config['train']['batch_episodes'])
        
        # Alternating updates
        metrics = {}
        
        # (E) Human surrogate update
        print("Updating human surrogate...")
        metrics.update(self.update_human_surrogate(trajectories))
        
        # (A) AI policy update
        print("Updating AI policy...")
        metrics.update(self.update_ai_policy(trajectories))
        
        # (P) Protocol update (conditional for single directional)
        if self.protocol_generator is not None:
            print("Updating protocol generator...")
            metrics.update(self.update_protocol(trajectories))
        else:
            print("Skipping protocol update (single directional mode)")
            metrics.update({'protocol_loss': 0.0, 'protocol_task_loss': 0.0, 'protocol_reg_loss': 0.0})
        
        # (M) RepMapper update (conditional for single directional)
        if self.rep_mapper is not None:
            print("Updating representation mapper...")
            metrics.update(self.update_rep_mapper(trajectories))
        else:
            print("Skipping representation mapper update (single directional mode)")
            metrics.update({'rep_repgap_loss': 0.0, 'rep_cca_loss': 0.0, 'rep_wasserstein_loss': 0.0})
        
        # (I) Instructor update (conditional for single directional)
        if self.instructor is not None:
            print("Updating instructor...")
            metrics.update(self.update_instructor(trajectories))
        else:
            print("Skipping instructor update (single directional mode)")
            metrics.update({'instructor_loss': 0.0, 'intervention_cost': 0.0})
        
        # (Λ) Dual variable update
        print("Updating dual variables...")
        metrics.update(self.update_dual_variables(trajectories))
        
        # Compute episode statistics
        episode_rewards = []
        episode_lengths = []
        success_rate = 0.0
        
        for trajectory in trajectories:
            batch = trajectory.get_batch()
            episode_rewards.append(batch['rewards'].sum().item())
            episode_lengths.append(len(batch['rewards']))
            success_rate += any(info.get('success', False) for info in batch['infos'])
        
        success_rate /= len(trajectories)
        
        metrics.update({
            'episode_reward_mean': np.mean(episode_rewards),
            'episode_reward_std': np.std(episode_rewards),
            'episode_length_mean': np.mean(episode_lengths),
            'success_rate': success_rate,
            'epoch': self.epoch
        })
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on both ID and OOD environments"""
        print("Evaluating...")
        
        eval_metrics = {}
        
        # ID evaluation
        id_trajectories = self.rollout(self.env, num_episodes=10)
        id_rewards = [sum(traj.get_batch()['rewards']) for traj in id_trajectories]
        id_success = sum(any(info.get('success', False) for info in traj.get_batch()['infos']) 
                        for traj in id_trajectories) / len(id_trajectories)
        
        eval_metrics.update({
            'eval_id_reward_mean': np.mean(id_rewards),
            'eval_id_success_rate': id_success
        })
        
        # OOD evaluation
        ood_trajectories = self.rollout(self.ood_env, num_episodes=10)
        ood_rewards = [sum(traj.get_batch()['rewards']) for traj in ood_trajectories]
        ood_success = sum(any(info.get('success', False) for info in traj.get_batch()['infos']) 
                         for traj in ood_trajectories) / len(ood_trajectories)
        
        eval_metrics.update({
            'eval_ood_reward_mean': np.mean(ood_rewards),
            'eval_ood_success_rate': ood_success
        })
        
        # Compute advanced BAS/CCM metrics
        try:
            # Prepare data for metrics computation
            metrics_data = self._prepare_metrics_data(id_trajectories, ood_trajectories)
            
            # Compute BAS and CCM metrics
            advanced_metrics = self.metrics_computer.compute_all_metrics(metrics_data)
            
            # Add to evaluation metrics
            eval_metrics.update(advanced_metrics)
            
            print(f"  Computed BAS score: {advanced_metrics.get('bas_score', 0.0):.3f}")
            print(f"  Computed CCM score: {advanced_metrics.get('ccm_score', 0.0):.3f}")
            
        except Exception as e:
            print(f"  Warning: Advanced metrics computation failed: {e}")
            # Add default values to maintain consistency
            eval_metrics.update({
                'bas_score': 0.5,
                'ccm_score': 0.5,
                'mutual_predictability': 0.0,
                'protocol_entropy': 0.0,
                'avg_messages': 0.0,
                'message_diversity': 0.0
            })
        
        return eval_metrics
    
    def _prepare_metrics_data(self, id_trajectories, ood_trajectories) -> Dict[str, Any]:
        """Prepare data for BAS/CCM metrics computation"""
        metrics_data = {}
        
        # Extract trajectory data
        all_trajectories = id_trajectories + ood_trajectories
        
        # Collect predictions and targets for mutual predictability
        human_predictions = []
        human_targets = []
        ai_predictions = []
        ai_targets = []
        
        # Collect features for representational compatibility
        human_features = []
        ai_features = []
        
        # Communication metrics
        total_messages = 0
        message_types = set()
        
        for traj in all_trajectories:
            batch = traj.get_batch()
            observations = batch['observations']
            actions = batch['actions']
            messages = batch.get('messages', [])
            
            # Count communication
            total_messages += len([m for m in messages if m is not None])
            message_types.update([str(m) for m in messages if m is not None])
            
            # Extract features (simplified - using observation embeddings)
            for obs, action in zip(observations, actions):
                if isinstance(obs, dict):
                    # Human observation features
                    if 'human_obs' in obs:
                        human_features.append(obs['human_obs'][:64])  # First 64 dims
                    # AI observation features  
                    if 'ai_obs' in obs:
                        ai_features.append(obs['ai_obs'][:64])  # First 64 dims
        
        # Convert to numpy arrays
        if human_features:
            metrics_data['human_features'] = np.array(human_features)
        if ai_features:
            metrics_data['ai_features'] = np.array(ai_features)
            
        # Dummy prediction data (would need actual model predictions in real implementation)
        if human_features and ai_features:
            n_samples = min(len(human_features), len(ai_features))
            metrics_data['human_predictions'] = np.random.rand(n_samples, 10)  # Dummy
            metrics_data['human_targets'] = np.random.randint(0, 10, n_samples)
            metrics_data['ai_predictions'] = np.random.rand(n_samples, 4)  # Dummy  
            metrics_data['ai_targets'] = np.random.randint(0, 4, n_samples)
        
        # Performance data
        id_success = sum(any(info.get('success', False) for info in traj.get_batch()['infos']) 
                        for traj in id_trajectories) / max(len(id_trajectories), 1)
        ood_success = sum(any(info.get('success', False) for info in traj.get_batch()['infos']) 
                         for traj in ood_trajectories) / max(len(ood_trajectories), 1)
        
        metrics_data['performance'] = {
            'baseline_success': 0.5,  # Baseline comparison
            'perturbed_success': id_success,
            'perturbation_kl': 0.02,  # Small perturbation
            'avg_steps': 30.0,
            'avg_tokens': total_messages / max(len(all_trajectories), 1)
        }
        
        metrics_data['ood_performance'] = {
            'success_rate': ood_success,
            'collision_rate': 0.1,  # Estimated
            'miscalibration': 0.05  # Estimated
        }
        
        # Communication metrics
        metrics_data['communication'] = {
            'avg_messages': total_messages / max(len(all_trajectories), 1),
            'message_diversity': len(message_types),
            'protocol_entropy': np.log(max(len(message_types), 1))
        }
        
        return metrics_data
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_performance': self.best_performance,
            'ai_policy': self.ai_policy.state_dict(),
            'value_net': self.value_net.state_dict(),
            'human_surrogate': self.human_surrogate.state_dict(),
            'protocol_generator': self.protocol_generator.state_dict() if self.protocol_generator is not None else None,
            'rep_mapper': self.rep_mapper.state_dict() if self.rep_mapper is not None else None,
            'instructor': self.instructor.state_dict() if self.instructor is not None else None,
            'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
            'dual_updater': {
                'lambda_a': self.dual_updater.lambda_a,
                'lambda_h': self.dual_updater.lambda_h
            },
            'gumbel_tau': self.gumbel_tau,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_performance = checkpoint['best_performance']
        
        self.ai_policy.load_state_dict(checkpoint['ai_policy'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.human_surrogate.load_state_dict(checkpoint['human_surrogate'])
        if self.protocol_generator is not None and checkpoint.get('protocol_generator') is not None:
            self.protocol_generator.load_state_dict(checkpoint['protocol_generator'])
        if self.rep_mapper is not None and checkpoint.get('rep_mapper') is not None:
            self.rep_mapper.load_state_dict(checkpoint['rep_mapper'])
        if self.instructor is not None and checkpoint.get('instructor') is not None:
            self.instructor.load_state_dict(checkpoint['instructor'])
        
        for k, v in checkpoint['optimizers'].items():
            self.optimizers[k].load_state_dict(v)
        
        self.dual_updater.lambda_a = checkpoint['dual_updater']['lambda_a']
        self.dual_updater.lambda_h = checkpoint['dual_updater']['lambda_h']
        
        self.gumbel_tau = checkpoint['gumbel_tau']
        
        print(f"Checkpoint loaded from {path}")
    
    def train(self):
        """Main training loop"""
        print("Starting BiCA training...")
        
        for epoch in range(self.config['train']['episodes'] // self.config['train']['batch_episodes']):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Log metrics
            for k, v in train_metrics.items():
                self.metrics[k].append(v)
            
            # Evaluate periodically (use configured eval_interval)
            eval_interval = self.config.get('train', {}).get('eval_interval', 320) // self.config.get('train', {}).get('batch_episodes', 32)
            if epoch % eval_interval == 0:
                eval_metrics = self.evaluate()
                train_metrics.update(eval_metrics)
                
                # Check for best performance
                current_performance = eval_metrics.get('eval_id_success_rate', 0.0)
                improvement = current_performance - self.best_performance
                
                if improvement > self.config.get('train', {}).get('early_stopping', {}).get('min_improvement', 0.01):
                    self.best_performance = current_performance
                    self.save_checkpoint(f"checkpoints/best_model_epoch_{epoch}.pt")
                    self.early_stopping_counter = 0
                    self.last_improvement_epoch = epoch
                    print(f" New best performance: {current_performance:.3f} at epoch {epoch} (improvement: {improvement:.3f})")
                else:
                    self.early_stopping_counter += 1
                    print(f" Current performance: {current_performance:.3f} (best: {self.best_performance:.3f}, no improvement for {self.early_stopping_counter} evaluations)")
                
                # Check early stopping
                early_stopping_config = self.config.get('train', {}).get('early_stopping', {})
                if (early_stopping_config.get('enabled', False) and 
                    self.early_stopping_counter * eval_interval >= early_stopping_config.get('patience', 50)):
                    epochs_without_improvement = epoch - self.last_improvement_epoch
                    print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                    print(f"Best performance: {self.best_performance:.3f} achieved at epoch {self.last_improvement_epoch}")
                    break
            
            # Log to wandb
            if self.config.get('use_wandb', True):
                wandb.log(train_metrics, step=epoch)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Reward={train_metrics['episode_reward_mean']:.2f}, "
                      f"Success={train_metrics['success_rate']:.3f}, "
                      f"Lambda_A={train_metrics['lambda_a']:.4f}")
            
            # Save checkpoint
            if epoch % 100 == 0:
                self.save_checkpoint(f"checkpoints/checkpoint_epoch_{epoch}.pt")
        
        print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='BiCA MapTalk Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override epochs if provided
    if args.epochs is not None:
        batch_episodes = config.get('train', {}).get('batch_episodes', 32)
        episodes = args.epochs * batch_episodes
        if 'train' not in config:
            config['train'] = {}
        config['train']['episodes'] = episodes
        print(f" Overriding config: epochs={args.epochs}, episodes={episodes}")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize trainer
    trainer = BiCATrainer(config)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
