"""
Latent Navigator for BiCA Latent-Navigator-Lite Experiment
Implements AI-Human cognitive transfer in learned latent space
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from collections import deque

from .vae_model import BetaVAE, ProjectionNetwork
from .data_loader import ScoringOracle


class LatentNavigator:
    """
    Main navigator for exploring latent space
    
    Combines AI suggestions with human interactions to explore
    and understand the latent space structure
    """
    
    def __init__(self,
                 vae_model: BetaVAE,
                 projection_network: ProjectionNetwork,
                 scoring_oracle: ScoringOracle,
                 config: Dict[str, Any]):
        self.vae_model = vae_model
        self.projection_network = projection_network
        self.scoring_oracle = scoring_oracle
        self.config = config
        
        # Navigation state
        self.current_position = np.array([0.0, 0.0])  # Current 2D position
        self.visited_positions = []
        self.scores_history = []
        self.suggestions_history = []
        
        # Exploration strategy
        self.exploration_strategy = config.get('exploration_strategy', 'gaussian_process')
        self.gp_regressor = self._init_gaussian_process()
        
        # Human interaction model
        try:
            self.human_model = HumanSurrogate(config.get('human_model', {}))
        except Exception as e:
            print(f"Warning: Failed to initialize HumanSurrogate: {e}")
            # Create a simple mock human model
            class MockHuman:
                def decide_next_click(self, ai_suggestions, current_state):
                    # Just return the first AI suggestion or a random point
                    if ai_suggestions:
                        return ai_suggestions[0]
                    return (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
            self.human_model = MockHuman()
        
        # Metrics tracking
        self.metrics = {
            'best_score': 0.0,
            'total_clicks': 0,
            'novelty_scores': [],
            'cognitive_gains': []
        }
        
        # Device
        try:
            self.device = next(vae_model.parameters()).device
        except (StopIteration, AttributeError):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _init_gaussian_process(self) -> GaussianProcessRegressor:
        """Initialize Gaussian Process for exploration"""
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.5)
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.1,
            n_restarts_optimizer=10,
            normalize_y=True
        )
    
    def suggest_region(self, 
                      strategy: str = 'uncertainty',
                      num_suggestions: int = 5) -> List[Tuple[float, float]]:
        """
        Suggest regions for exploration
        
        Args:
            strategy: Suggestion strategy ('uncertainty', 'expected_improvement', 'random')
            num_suggestions: Number of suggestions to generate
            
        Returns:
            suggestions: List of (x, y) coordinates to explore
        """
        if strategy == 'uncertainty' and len(self.visited_positions) > 3:
            return self._suggest_uncertainty_based(num_suggestions)
        elif strategy == 'expected_improvement' and len(self.visited_positions) > 3:
            return self._suggest_expected_improvement(num_suggestions)
        else:
            return self._suggest_random(num_suggestions)
    
    def _suggest_uncertainty_based(self, num_suggestions: int) -> List[Tuple[float, float]]:
        """Suggest regions with high uncertainty"""
        # Generate candidate points
        candidates = self._generate_candidate_points(1000)
        
        # Predict with GP
        if len(self.visited_positions) > 0:
            X = np.array(self.visited_positions)
            y = np.array(self.scores_history)
            
            self.gp_regressor.fit(X, y)
            
            # Get uncertainty (std) for candidates
            _, std = self.gp_regressor.predict(candidates, return_std=True)
            
            # Select points with highest uncertainty
            top_indices = np.argsort(std)[-num_suggestions:]
            suggestions = [tuple(candidates[i]) for i in top_indices]
        else:
            suggestions = self._suggest_random(num_suggestions)
        
        return suggestions
    
    def _suggest_expected_improvement(self, num_suggestions: int) -> List[Tuple[float, float]]:
        """Suggest regions with high expected improvement"""
        candidates = self._generate_candidate_points(1000)
        
        if len(self.visited_positions) > 0:
            X = np.array(self.visited_positions)
            y = np.array(self.scores_history)
            
            self.gp_regressor.fit(X, y)
            
            # Compute expected improvement
            best_score = max(self.scores_history)
            mean, std = self.gp_regressor.predict(candidates, return_std=True)
            
            # Expected improvement calculation
            improvement = mean - best_score
            Z = improvement / (std + 1e-9)
            
            from scipy.stats import norm
            ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
            
            # Select points with highest EI
            top_indices = np.argsort(ei)[-num_suggestions:]
            suggestions = [tuple(candidates[i]) for i in top_indices]
        else:
            suggestions = self._suggest_random(num_suggestions)
        
        return suggestions
    
    def _suggest_random(self, num_suggestions: int) -> List[Tuple[float, float]]:
        """Generate random suggestions"""
        suggestions = []
        for _ in range(num_suggestions):
            x = np.random.uniform(-1.0, 1.0)
            y = np.random.uniform(-1.0, 1.0)
            suggestions.append((x, y))
        return suggestions
    
    def _generate_candidate_points(self, num_points: int) -> np.ndarray:
        """Generate candidate points for exploration"""
        return np.random.uniform(-1.0, 1.0, size=(num_points, 2))
    
    def human_click(self, position: Tuple[float, float]) -> Dict[str, Any]:
        """
        Process human click at given position
        
        Args:
            position: (x, y) coordinates of click
            
        Returns:
            result: Dictionary with decoded sample and score
        """
        x, y = position
        self.current_position = np.array([x, y])
        
        # Convert 2D position back to latent space
        latent_sample = self._inverse_project(np.array([[x, y]]))
        
        # Decode to image
        with torch.no_grad():
            decoded_image = self.vae_model.decode(latent_sample)
            
            # Compute score using oracle
            # First encode the decoded image to get factors
            vae_outputs = self.vae_model(decoded_image)
            estimated_factors = self._estimate_factors_from_latent(vae_outputs['z'])
            score = self.scoring_oracle.compute_score(estimated_factors).item()
        
        # Update history
        self.visited_positions.append((x, y))
        self.scores_history.append(score)
        self.metrics['total_clicks'] += 1
        self.metrics['best_score'] = max(self.metrics['best_score'], score)
        
        # Compute novelty (distance to previous points)
        novelty = self._compute_novelty(position)
        self.metrics['novelty_scores'].append(novelty)
        
        result = {
            'position': position,
            'decoded_image': decoded_image.cpu(),
            'latent_representation': latent_sample.cpu(),
            'estimated_factors': estimated_factors.cpu(),
            'score': score,
            'novelty': novelty,
            'is_best': score == self.metrics['best_score']
        }
        
        return result
    
    def _inverse_project(self, positions_2d: np.ndarray) -> torch.Tensor:
        """
        Inverse projection from 2D back to latent space
        
        This is approximate since projection is not invertible.
        We use optimization to find latent codes that project close to target.
        """
        positions_2d = torch.from_numpy(positions_2d).float().to(self.device)
        batch_size = positions_2d.size(0)
        
        # Initialize latent codes
        latent_codes = torch.randn(batch_size, self.vae_model.latent_dim, 
                                  device=self.device, requires_grad=True)
        
        # Optimize to match 2D projection
        optimizer = optim.Adam([latent_codes], lr=0.01)
        
        for _ in range(100):  # Quick optimization
            optimizer.zero_grad()
            
            projected = self.projection_network(latent_codes)
            loss = torch.nn.functional.mse_loss(projected, positions_2d)
            
            loss.backward()
            optimizer.step()
            
            if loss.item() < 1e-4:
                break
        
        return latent_codes.detach()
    
    def _estimate_factors_from_latent(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """
        Estimate ground truth factors from latent codes
        
        This would ideally use a trained factor predictor,
        but for simplicity we use a linear approximation
        """
        # Simple linear mapping (would be learned in practice)
        # Assume first few latent dimensions correspond to main factors
        num_factors = 5  # shape, size, rotation, pos_x, pos_y
        
        factors = torch.zeros(latent_codes.size(0), num_factors, device=latent_codes.device)
        
        # Rough mapping (this would be learned from data)
        factors[:, 0] = torch.sigmoid(latent_codes[:, 0]) * 3  # shape (0-3)
        factors[:, 1] = torch.sigmoid(latent_codes[:, 1]) * 0.8 + 0.2  # size (0.2-1.0)
        factors[:, 2] = torch.sigmoid(latent_codes[:, 2]) * 2 * np.pi  # rotation (0-2π)
        factors[:, 3] = torch.sigmoid(latent_codes[:, 3])  # pos_x (0-1)
        factors[:, 4] = torch.sigmoid(latent_codes[:, 4])  # pos_y (0-1)
        
        return factors
    
    def _compute_novelty(self, position: Tuple[float, float]) -> float:
        """Compute novelty of position based on distance to previous points"""
        if len(self.visited_positions) <= 1:
            return 1.0
        
        pos = np.array(position)
        previous_positions = np.array(self.visited_positions[:-1])  # Exclude current
        
        # Minimum distance to previous points
        distances = np.linalg.norm(previous_positions - pos, axis=1)
        min_distance = np.min(distances)
        
        # Novelty score (higher for more distant points)
        novelty = np.tanh(min_distance * 2.0)  # Scale and saturate
        
        return novelty
    
    def get_exploration_map(self, grid_size: int = 50) -> Dict[str, np.ndarray]:
        """
        Generate exploration map showing predicted scores and uncertainty
        
        Args:
            grid_size: Resolution of the map
            
        Returns:
            exploration_map: Dictionary with score and uncertainty maps
        """
        # Generate grid
        x = np.linspace(-1.0, 1.0, grid_size)
        y = np.linspace(-1.0, 1.0, grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        if len(self.visited_positions) > 3:
            # Fit GP and predict
            X = np.array(self.visited_positions)
            y_scores = np.array(self.scores_history)
            
            self.gp_regressor.fit(X, y_scores)
            
            mean_pred, std_pred = self.gp_regressor.predict(grid_points, return_std=True)
            
            score_map = mean_pred.reshape(grid_size, grid_size)
            uncertainty_map = std_pred.reshape(grid_size, grid_size)
        else:
            # No predictions yet
            score_map = np.zeros((grid_size, grid_size))
            uncertainty_map = np.ones((grid_size, grid_size))
        
        return {
            'score_map': score_map,
            'uncertainty_map': uncertainty_map,
            'x_grid': xx,
            'y_grid': yy
        }
    
    def compute_cognitive_gain(self, 
                              pre_quiz_accuracy: float,
                              post_quiz_accuracy: float,
                              pre_quiz_ece: float,
                              post_quiz_ece: float) -> Dict[str, float]:
        """
        Compute cognitive gain metrics
        
        Args:
            pre_quiz_accuracy: Accuracy before navigation
            post_quiz_accuracy: Accuracy after navigation
            pre_quiz_ece: ECE before navigation
            post_quiz_ece: ECE after navigation
            
        Returns:
            cognitive_gains: Dictionary of cognitive gain metrics
        """
        accuracy_gain = post_quiz_accuracy - pre_quiz_accuracy
        ece_improvement = pre_quiz_ece - post_quiz_ece  # Lower ECE is better
        
        # Combined cognitive gain
        cognitive_gain = 0.7 * accuracy_gain + 0.3 * ece_improvement
        
        gains = {
            'accuracy_gain': accuracy_gain,
            'ece_improvement': ece_improvement,
            'cognitive_gain': cognitive_gain,
            'relative_accuracy_gain': accuracy_gain / max(pre_quiz_accuracy, 0.01)
        }
        
        self.metrics['cognitive_gains'].append(gains)
        
        return gains
    
    def get_navigation_summary(self) -> Dict[str, Any]:
        """Get summary of navigation session"""
        summary = {
            'total_clicks': self.metrics['total_clicks'],
            'best_score': self.metrics['best_score'],
            'average_score': np.mean(self.scores_history) if self.scores_history else 0.0,
            'average_novelty': np.mean(self.metrics['novelty_scores']) if self.metrics['novelty_scores'] else 0.0,
            'exploration_efficiency': self.metrics['best_score'] / max(self.metrics['total_clicks'], 1),
            'visited_positions': self.visited_positions.copy(),
            'scores_history': self.scores_history.copy()
        }
        
        if self.metrics['cognitive_gains']:
            latest_gains = self.metrics['cognitive_gains'][-1]
            summary.update(latest_gains)
        
        return summary


class HumanSurrogate:
    """
    Simple human surrogate for automated evaluation
    
    Models human clicking behavior with some exploration strategy
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exploration_noise = config.get('exploration_noise', 0.1)
        self.exploitation_weight = config.get('exploitation_weight', 0.7)
        self.memory_length = config.get('memory_length', 10)
        
        # Internal state
        self.click_history = deque(maxlen=self.memory_length)
    
    def decide_click(self, 
                    suggestions: List[Tuple[float, float]],
                    exploration_map: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Decide where to click based on suggestions and current knowledge
        
        Args:
            suggestions: AI suggestions
            exploration_map: Current exploration map
            
        Returns:
            click_position: Chosen (x, y) position
        """
        # Combine exploitation (high score areas) and exploration (suggestions)
        if np.random.random() < self.exploitation_weight and len(self.click_history) > 3:
            # Exploitation: click near high-scoring areas
            click_pos = self._exploit_good_regions(exploration_map)
        else:
            # Exploration: follow AI suggestions with some noise
            if suggestions:
                suggestion = suggestions[np.random.randint(len(suggestions))]
                noise_x = np.random.normal(0, self.exploration_noise)
                noise_y = np.random.normal(0, self.exploration_noise)
                
                click_pos = (
                    np.clip(suggestion[0] + noise_x, -1.0, 1.0),
                    np.clip(suggestion[1] + noise_y, -1.0, 1.0)
                )
            else:
                # Random exploration
                click_pos = (
                    np.random.uniform(-1.0, 1.0),
                    np.random.uniform(-1.0, 1.0)
                )
        
        self.click_history.append(click_pos)
        return click_pos
    
    def _exploit_good_regions(self, exploration_map: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """Click near regions with high predicted scores"""
        score_map = exploration_map['score_map']
        x_grid = exploration_map['x_grid']
        y_grid = exploration_map['y_grid']
        
        # Find high-score regions
        flat_scores = score_map.flatten()
        top_indices = np.argsort(flat_scores)[-10:]  # Top 10 regions
        
        # Select one randomly
        selected_idx = np.random.choice(top_indices)
        
        # Convert back to coordinates
        i, j = np.unravel_index(selected_idx, score_map.shape)
        
        # Add some noise
        base_x = x_grid[i, j]
        base_y = y_grid[i, j]
        
        noise_x = np.random.normal(0, self.exploration_noise)
        noise_y = np.random.normal(0, self.exploration_noise)
        
        click_x = np.clip(base_x + noise_x, -1.0, 1.0)
        click_y = np.clip(base_y + noise_y, -1.0, 1.0)
        
        return (click_x, click_y)


def create_latent_navigator(config: Dict[str, Any]) -> LatentNavigator:
    """Factory function to create latent navigator"""
    from .vae_model import create_beta_vae, create_projection_network
    from .data_loader import DatasetFactory
    
    # Create models
    vae_model = create_beta_vae(config['vae'])
    projection_network = create_projection_network(config['projection'])
    
    # Create scoring oracle
    dataset_name = config.get('dataset_name', 'geometric_shapes')
    scoring_oracle = DatasetFactory.create_scoring_oracle(
        dataset_name, **config.get('oracle_kwargs', {})
    )
    
    # Create navigator
    navigator = LatentNavigator(
        vae_model, projection_network, scoring_oracle, config
    )
    
    return navigator
