"""
MapTalk Environment: 8x8 gridworld with bidirectional communication
Implements the main experimental environment from the BiCA paper
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import gymnasium as gym
from gymnasium import spaces


class MapTalkEnv(gym.Env):
    """
    8x8 gridworld environment for BiCA MapTalk experiment.
    
    Features:
    - AI agent has egocentric 3x3 patch observation + heading
    - Human has full map view
    - Bidirectional messaging between AI and human
    - Instructor interventions
    - Configurable obstacle rates and rewards
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Environment parameters
        self.grid_size = config.get('grid_size', 8)
        self.obstacle_rate = config.get('obstacle_rate', [0.2, 0.3])
        self.max_steps = config.get('max_steps', 60)
        
        # Reward parameters
        self.reward_step = config.get('reward_step', -1.0)
        self.reward_collision = config.get('reward_collision', -5.0)
        self.reward_success = config.get('reward_success', 50.0)
        self.reward_token_cost = config.get('reward_token_cost', -0.05)
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # FWD, LEFT, RIGHT, STAY
        
        # AI observation: 3x3 patch + heading (4 directions)
        self.ai_obs_space = spaces.Box(
            low=0, high=1, shape=(3, 3, 2), dtype=np.float32  # walls + objects
        )
        self.ai_heading_space = spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32  # one-hot heading
        )
        
        # Human observation: full map
        self.human_obs_space = spaces.Box(
            low=0, high=1, shape=(self.grid_size, self.grid_size, 3), dtype=np.float32
        )
        
        # Message spaces (simplified as discrete for now)
        self.ai_message_space = spaces.Discrete(64)  # AI protocol messages
        self.human_message_space = spaces.Discrete(32)  # Human hint messages
        self.instructor_space = spaces.Discrete(8)  # Instructor interventions
        
        # Internal state
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.agent_heading = 0  # 0: North, 1: East, 2: South, 3: West
        self.step_count = 0
        self.done = False
        
        # Communication history
        self.message_history = deque(maxlen=10)
        
        # Directions for movement
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observations"""
        self.step_count = 0
        self.done = False
        self.message_history.clear()
        
        # Generate new grid
        self._generate_grid()
        
        # Return initial observations
        return {
            'ai_obs': self._get_ai_observation(),
            'human_obs': self._get_human_observation(),
            'ai_heading': self._get_heading_onehot(),
        }
    
    def step(self, action: int, ai_message: int = 0, human_message: int = 0, 
             instructor_action: int = 0) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: AI agent action (0: FWD, 1: LEFT, 2: RIGHT, 3: STAY)
            ai_message: AI protocol message
            human_message: Human hint message  
            instructor_action: Instructor intervention
            
        Returns:
            observations, reward, done, info
        """
        if self.done:
            return self._get_observations(), 0.0, True, {}
        
        self.step_count += 1
        
        # Store messages in history
        self.message_history.append({
            'ai_message': ai_message,
            'human_message': human_message,
            'instructor_action': instructor_action,
            'step': self.step_count
        })
        
        # Execute action
        reward = 0.0
        collision = False
        
        if action == 0:  # Forward
            new_pos = self._get_forward_position()
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
            else:
                collision = True
                reward += self.reward_collision
        elif action == 1:  # Turn left
            self.agent_heading = (self.agent_heading - 1) % 4
        elif action == 2:  # Turn right
            self.agent_heading = (self.agent_heading + 1) % 4
        elif action == 3:  # Stay
            pass
        
        # Calculate rewards
        reward += self.reward_step
        
        # Token cost for messages (simplified)
        if ai_message > 0 or human_message > 0:
            reward += self.reward_token_cost
        
        # Check if goal reached
        success = (self.agent_pos == self.goal_pos).all()
        if success:
            reward += self.reward_success
            self.done = True
        
        # Check if max steps reached
        if self.step_count >= self.max_steps:
            self.done = True
        
        info = {
            'success': success,
            'collision': collision,
            'step_count': self.step_count,
            'agent_pos': self.agent_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'messages': dict(self.message_history[-1]) if self.message_history else {}
        }
        
        return self._get_observations(), reward, self.done, info
    
    def _generate_grid(self):
        """Generate a new random grid with obstacles, start, and goal"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Add obstacles
        obs_rate = np.random.uniform(self.obstacle_rate[0], self.obstacle_rate[1])
        obstacle_mask = np.random.random((self.grid_size, self.grid_size)) < obs_rate
        self.grid[obstacle_mask] = 1  # 1 = obstacle
        
        # Place start and goal positions
        free_positions = np.argwhere(self.grid == 0)
        
        if len(free_positions) < 2:
            # Fallback: ensure at least start and goal are free
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            free_positions = np.argwhere(self.grid == 0)
        
        # Sample start and goal
        indices = np.random.choice(len(free_positions), size=2, replace=False)
        self.agent_pos = free_positions[indices[0]]
        self.goal_pos = free_positions[indices[1]]
        
        # Ensure reachability using BFS
        if not self._is_reachable():
            # Clear path between start and goal
            self._clear_path()
        
        # Reset heading
        self.agent_heading = np.random.randint(4)
    
    def _is_reachable(self) -> bool:
        """Check if goal is reachable from start using BFS"""
        queue = deque([tuple(self.agent_pos)])
        visited = set([tuple(self.agent_pos)])
        
        while queue:
            pos = queue.popleft()
            if pos == tuple(self.goal_pos):
                return True
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if (0 <= new_pos[0] < self.grid_size and 
                    0 <= new_pos[1] < self.grid_size and
                    new_pos not in visited and
                    self.grid[new_pos] == 0):
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return False
    
    def _clear_path(self):
        """Clear a simple path between start and goal"""
        # Simple path clearing - can be improved
        x1, y1 = self.agent_pos
        x2, y2 = self.goal_pos
        
        # Clear horizontal path
        for x in range(min(x1, x2), max(x1, x2) + 1):
            self.grid[x, y1] = 0
        
        # Clear vertical path
        for y in range(min(y1, y2), max(y1, y2) + 1):
            self.grid[x2, y] = 0
    
    def _get_forward_position(self) -> np.ndarray:
        """Get position if agent moves forward"""
        dx, dy = self.directions[self.agent_heading]
        return self.agent_pos + np.array([dx, dy])
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if position is valid (in bounds and not obstacle)"""
        x, y = pos
        return (0 <= x < self.grid_size and 
                0 <= y < self.grid_size and 
                self.grid[x, y] == 0)
    
    def _get_ai_observation(self) -> np.ndarray:
        """Get AI agent's egocentric 3x3 patch observation"""
        obs = np.zeros((3, 3, 2), dtype=np.float32)
        
        agent_x, agent_y = self.agent_pos
        
        # Get 3x3 patch around agent
        for i in range(3):
            for j in range(3):
                # Transform to world coordinates based on heading
                local_x, local_y = i - 1, j - 1  # Center at (1,1)
                
                # Rotate based on heading
                if self.agent_heading == 0:  # North
                    world_x, world_y = agent_x + local_x, agent_y + local_y
                elif self.agent_heading == 1:  # East
                    world_x, world_y = agent_x - local_y, agent_y + local_x
                elif self.agent_heading == 2:  # South
                    world_x, world_y = agent_x - local_x, agent_y - local_y
                else:  # West
                    world_x, world_y = agent_x + local_y, agent_y - local_x
                
                # Check bounds and set observation
                if (0 <= world_x < self.grid_size and 
                    0 <= world_y < self.grid_size):
                    obs[i, j, 0] = self.grid[world_x, world_y]  # Walls/obstacles
                    if (world_x, world_y) == tuple(self.goal_pos):
                        obs[i, j, 1] = 1.0  # Goal marker
                else:
                    obs[i, j, 0] = 1.0  # Out of bounds = wall
        
        return obs
    
    def _get_human_observation(self) -> np.ndarray:
        """Get human's full map observation"""
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # Channel 0: walls/obstacles
        obs[:, :, 0] = self.grid
        
        # Channel 1: agent position
        obs[self.agent_pos[0], self.agent_pos[1], 1] = 1.0
        
        # Channel 2: goal position
        obs[self.goal_pos[0], self.goal_pos[1], 2] = 1.0
        
        return obs
    
    def _get_heading_onehot(self) -> np.ndarray:
        """Get agent heading as one-hot vector"""
        heading = np.zeros(4, dtype=np.float32)
        heading[self.agent_heading] = 1.0
        return heading
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get all observations"""
        return {
            'ai_obs': self._get_ai_observation(),
            'human_obs': self._get_human_observation(),
            'ai_heading': self._get_heading_onehot(),
        }
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.step_count}")
            print(f"Agent: {self.agent_pos}, Heading: {self.agent_heading}")
            print(f"Goal: {self.goal_pos}")
            
            # Print grid
            display_grid = self.grid.copy().astype(str)
            display_grid[display_grid == '0'] = '.'
            display_grid[display_grid == '1'] = '#'
            display_grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
            display_grid[self.goal_pos[0], self.goal_pos[1]] = 'G'
            
            for row in display_grid:
                print(' '.join(row))
            print()


class OODMapTalkEnv(MapTalkEnv):
    """
    Out-of-distribution version of MapTalk environment.
    
    Features additional challenges:
    - Higher obstacle rates
    - Sensor noise
    - Structural patterns (corridors, rooms)
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Modify config for OOD settings
        ood_config = config.copy()
        ood_config['obstacle_rate'] = config.get('ood_obstacle_rate', [0.35, 0.45])
        
        super().__init__(ood_config)
        
        self.sensor_flip_prob = config.get('sensor_flip_prob', 0.1)
        self.ood_patterns = config.get('ood_patterns', ['corridor', 'rooms'])
        self.current_pattern = None
    
    def _generate_grid(self):
        """Generate OOD grid with structural patterns"""
        # Choose pattern
        if self.ood_patterns:
            self.current_pattern = np.random.choice(self.ood_patterns)
        else:
            self.current_pattern = 'random'
        
        if self.current_pattern == 'corridor':
            self._generate_corridor_grid()
        elif self.current_pattern == 'rooms':
            self._generate_rooms_grid()
        else:
            super()._generate_grid()
    
    def _generate_corridor_grid(self):
        """Generate grid with corridor pattern"""
        self.grid = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Create main corridor
        corridor_width = 2
        mid = self.grid_size // 2
        
        # Horizontal corridor
        self.grid[mid-corridor_width//2:mid+corridor_width//2+1, :] = 0
        
        # Vertical corridors
        for i in range(0, self.grid_size, 3):
            if i < self.grid_size:
                self.grid[:, i] = 0
        
        # Place start and goal
        free_positions = np.argwhere(self.grid == 0)
        if len(free_positions) >= 2:
            indices = np.random.choice(len(free_positions), size=2, replace=False)
            self.agent_pos = free_positions[indices[0]]
            self.goal_pos = free_positions[indices[1]]
        else:
            # Fallback
            self.agent_pos = np.array([mid, 0])
            self.goal_pos = np.array([mid, self.grid_size-1])
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 0
            self.grid[self.goal_pos[0], self.goal_pos[1]] = 0
        
        self.agent_heading = np.random.randint(4)
    
    def _generate_rooms_grid(self):
        """Generate grid with room pattern"""
        self.grid = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Create 4 rooms
        room_size = self.grid_size // 2 - 1
        
        # Room positions
        rooms = [
            (1, 1, room_size, room_size),  # Top-left
            (1, self.grid_size//2+1, room_size, room_size),  # Top-right
            (self.grid_size//2+1, 1, room_size, room_size),  # Bottom-left
            (self.grid_size//2+1, self.grid_size//2+1, room_size, room_size)  # Bottom-right
        ]
        
        # Clear rooms
        for x, y, w, h in rooms:
            self.grid[x:x+w, y:y+h] = 0
        
        # Add doorways
        mid = self.grid_size // 2
        self.grid[mid, mid] = 0  # Center connection
        self.grid[mid-1, mid] = 0
        self.grid[mid, mid-1] = 0
        
        # Place start and goal in different rooms
        room_centers = [(room_size//2+1, room_size//2+1), 
                       (room_size//2+1, self.grid_size//2+1+room_size//2),
                       (self.grid_size//2+1+room_size//2, room_size//2+1),
                       (self.grid_size//2+1+room_size//2, self.grid_size//2+1+room_size//2)]
        
        start_room, goal_room = np.random.choice(4, size=2, replace=False)
        self.agent_pos = np.array(room_centers[start_room])
        self.goal_pos = np.array(room_centers[goal_room])
        
        self.agent_heading = np.random.randint(4)
    
    def _get_ai_observation(self) -> np.ndarray:
        """Get AI observation with sensor noise"""
        obs = super()._get_ai_observation()
        
        # Add sensor noise
        if self.sensor_flip_prob > 0:
            noise_mask = np.random.random(obs.shape) < self.sensor_flip_prob
            obs[noise_mask] = 1.0 - obs[noise_mask]
        
        return obs


def create_env(config: Dict[str, Any]) -> MapTalkEnv:
    """Factory function to create environment"""
    if config.get('ood', False):
        return OODMapTalkEnv(config)
    else:
        return MapTalkEnv(config)
