"""
UI Interface for Latent-Navigator-Lite
Provides interactive interface for human navigation of latent space
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from typing import Dict, List, Tuple, Optional, Any, Callable
import threading
import queue

from .navigator import LatentNavigator


class NavigatorUI:
    """
    Interactive UI for latent space navigation
    
    Provides:
    - 2D visualization of latent space
    - Click interface for exploration
    - AI suggestion display
    - Score feedback
    - Progress tracking
    """
    
    def __init__(self, 
                 navigator: LatentNavigator,
                 config: Dict[str, Any]):
        self.navigator = navigator
        self.config = config
        
        # UI state
        self.current_image = None
        self.suggestion_points = []
        self.click_history = []
        
        # Threading for UI responsiveness
        self.ui_queue = queue.Queue()
        self.is_running = True
        
        # Initialize UI
        self._init_matplotlib_ui()
        self._init_tkinter_ui()
    
    def _init_matplotlib_ui(self):
        """Initialize matplotlib-based 2D exploration interface"""
        self.fig, (self.ax_map, self.ax_image) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 2D exploration map
        self.ax_map.set_xlim(-1.1, 1.1)
        self.ax_map.set_ylim(-1.1, 1.1)
        self.ax_map.set_xlabel('Latent Dimension 1')
        self.ax_map.set_ylabel('Latent Dimension 2')
        self.ax_map.set_title('Latent Space Exploration')
        self.ax_map.grid(True, alpha=0.3)
        
        # Image display
        self.ax_image.set_title('Decoded Sample')
        self.ax_image.axis('off')
        
        # Connect click events
        self.fig.canvas.mpl_connect('button_press_event', self._on_matplotlib_click)
        
        # Initialize empty plots
        self.scatter_visited = self.ax_map.scatter([], [], c=[], s=50, cmap='viridis', 
                                                  alpha=0.7, label='Visited')
        self.scatter_suggestions = self.ax_map.scatter([], [], c='red', s=100, 
                                                      marker='x', label='AI Suggestions')
        self.ax_map.legend()
        
        # Add control buttons
        self._add_matplotlib_controls()
    
    def _add_matplotlib_controls(self):
        """Add control buttons to matplotlib interface"""
        # Suggestion button
        ax_suggest = plt.axes([0.02, 0.02, 0.1, 0.05])
        self.btn_suggest = Button(ax_suggest, 'Get Suggestions')
        self.btn_suggest.on_clicked(self._get_ai_suggestions)
        
        # Clear button
        ax_clear = plt.axes([0.14, 0.02, 0.08, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self._clear_exploration)
        
        # Strategy selector (simplified)
        ax_strategy = plt.axes([0.24, 0.02, 0.15, 0.05])
        self.btn_strategy = Button(ax_strategy, 'Toggle Strategy')
        self.btn_strategy.on_clicked(self._toggle_strategy)
        
        self.current_strategy = 'uncertainty'
    
    def _init_tkinter_ui(self):
        """Initialize tkinter-based control panel"""
        self.root = tk.Tk()
        self.root.title("Latent Navigator Control Panel")
        self.root.geometry("400x600")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Statistics display
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="5")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.stats_labels = {}
        stats_metrics = ['Total Clicks', 'Best Score', 'Current Score', 'Avg Score', 'Coverage']
        
        for i, metric in enumerate(stats_metrics):
            ttk.Label(stats_frame, text=f"{metric}:").grid(row=i, column=0, sticky=tk.W)
            label = ttk.Label(stats_frame, text="0.000")
            label.grid(row=i, column=1, sticky=tk.E)
            self.stats_labels[metric] = label
        
        # Progress display
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.progress_label = ttk.Label(progress_frame, text="Click to start exploring!")
        self.progress_label.grid(row=1, column=0, pady=2)
        
        # Controls
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        controls_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Strategy selection
        ttk.Label(controls_frame, text="Strategy:").grid(row=0, column=0, sticky=tk.W)
        self.strategy_var = tk.StringVar(value="uncertainty")
        strategy_combo = ttk.Combobox(controls_frame, textvariable=self.strategy_var,
                                    values=["uncertainty", "expected_improvement", "random"])
        strategy_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Number of suggestions
        ttk.Label(controls_frame, text="Suggestions:").grid(row=1, column=0, sticky=tk.W)
        self.suggestions_var = tk.IntVar(value=5)
        suggestions_spin = ttk.Spinbox(controls_frame, from_=1, to=10, 
                                     textvariable=self.suggestions_var, width=10)
        suggestions_spin.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Get AI Suggestions", 
                  command=self._get_ai_suggestions).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Auto Explore", 
                  command=self._auto_explore).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Reset", 
                  command=self._reset_exploration).pack(side=tk.LEFT, padx=2)
        
        # Recent clicks display
        clicks_frame = ttk.LabelFrame(main_frame, text="Recent Clicks", padding="5")
        clicks_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Scrollable text area for click history
        self.clicks_text = tk.Text(clicks_frame, height=10, width=40)
        scrollbar = ttk.Scrollbar(clicks_frame, orient="vertical", command=self.clicks_text.yview)
        self.clicks_text.configure(yscrollcommand=scrollbar.set)
        
        self.clicks_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        clicks_frame.columnconfigure(0, weight=1)
        clicks_frame.rowconfigure(0, weight=1)
        
        # Start UI update thread
        self._start_ui_thread()
    
    def _start_ui_thread(self):
        """Start thread for UI updates"""
        def ui_update_loop():
            while self.is_running:
                try:
                    # Process UI updates
                    while not self.ui_queue.empty():
                        update_func = self.ui_queue.get_nowait()
                        update_func()
                    
                    # Update statistics
                    self._update_statistics()
                    
                    # Sleep briefly
                    threading.Event().wait(0.1)
                    
                except Exception as e:
                    print(f"UI update error: {e}")
        
        self.ui_thread = threading.Thread(target=ui_update_loop, daemon=True)
        self.ui_thread.start()
    
    def _on_matplotlib_click(self, event):
        """Handle click events on matplotlib plot"""
        if event.inaxes == self.ax_map and event.button == 1:  # Left click
            x, y = event.xdata, event.ydata
            
            if x is not None and y is not None:
                # Clamp to valid range
                x = np.clip(x, -1.0, 1.0)
                y = np.clip(y, -1.0, 1.0)
                
                # Process click
                self._process_click((x, y))
    
    def _process_click(self, position: Tuple[float, float]):
        """Process a click at given position"""
        try:
            # Execute navigation
            result = self.navigator.human_click(position)
            
            # Update UI
            self.ui_queue.put(lambda: self._update_after_click(result))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process click: {e}")
    
    def _update_after_click(self, result: Dict[str, Any]):
        """Update UI after processing a click"""
        # Update matplotlib plot
        self._update_exploration_plot()
        
        # Update image display
        self._update_image_display(result['decoded_image'])
        
        # Update click history
        self._update_click_history(result)
        
        # Refresh matplotlib
        self.fig.canvas.draw()
    
    def _update_exploration_plot(self):
        """Update the 2D exploration plot"""
        if not self.navigator.visited_positions:
            return
        
        # Get positions and scores
        positions = np.array(self.navigator.visited_positions)
        scores = np.array(self.navigator.scores_history)
        
        # Update visited points scatter plot
        self.scatter_visited.set_offsets(positions)
        self.scatter_visited.set_array(scores)
        
        # Update suggestions if available
        if self.suggestion_points:
            suggestion_positions = np.array(self.suggestion_points)
            self.scatter_suggestions.set_offsets(suggestion_positions)
        
        # Update exploration map background (optional)
        if len(self.navigator.visited_positions) > 3:
            self._update_exploration_background()
    
    def _update_exploration_background(self):
        """Update background heatmap showing predicted scores"""
        try:
            exploration_map = self.navigator.get_exploration_map(grid_size=30)
            
            # Clear previous background
            for collection in self.ax_map.collections[:-2]:  # Keep scatter plots
                collection.remove()
            
            # Add heatmap
            im = self.ax_map.imshow(
                exploration_map['score_map'],
                extent=[-1, 1, -1, 1],
                origin='lower',
                alpha=0.3,
                cmap='viridis'
            )
            
        except Exception as e:
            print(f"Failed to update exploration background: {e}")
    
    def _update_image_display(self, decoded_image: torch.Tensor):
        """Update the decoded image display"""
        try:
            # Convert tensor to numpy
            if decoded_image.dim() == 4:
                image_np = decoded_image[0].detach().cpu().numpy()
            else:
                image_np = decoded_image.detach().cpu().numpy()
            
            # Handle different image formats
            if image_np.shape[0] == 1:  # Grayscale
                image_np = image_np[0]
                cmap = 'gray'
            else:  # RGB
                image_np = np.transpose(image_np, (1, 2, 0))
                cmap = None
            
            # Display image
            self.ax_image.clear()
            self.ax_image.imshow(image_np, cmap=cmap)
            self.ax_image.set_title('Decoded Sample')
            self.ax_image.axis('off')
            
        except Exception as e:
            print(f"Failed to update image display: {e}")
    
    def _update_click_history(self, result: Dict[str, Any]):
        """Update click history display"""
        try:
            position = result['position']
            score = result['score']
            novelty = result['novelty']
            
            # Format click information
            click_info = f"Click {len(self.navigator.visited_positions)}: "
            click_info += f"({position[0]:.3f}, {position[1]:.3f}) "
            click_info += f"Score: {score:.3f} "
            click_info += f"Novelty: {novelty:.3f}"
            
            if result.get('is_best', False):
                click_info += " ⭐ NEW BEST!"
            
            click_info += "\n"
            
            # Add to text widget
            self.clicks_text.insert(tk.END, click_info)
            self.clicks_text.see(tk.END)
            
        except Exception as e:
            print(f"Failed to update click history: {e}")
    
    def _update_statistics(self):
        """Update statistics display"""
        try:
            summary = self.navigator.get_navigation_summary()
            
            # Update labels
            self.stats_labels['Total Clicks'].config(text=str(summary['total_clicks']))
            self.stats_labels['Best Score'].config(text=f"{summary['best_score']:.3f}")
            
            if summary['scores_history']:
                current_score = summary['scores_history'][-1]
                avg_score = summary['average_score']
            else:
                current_score = 0.0
                avg_score = 0.0
            
            self.stats_labels['Current Score'].config(text=f"{current_score:.3f}")
            self.stats_labels['Avg Score'].config(text=f"{avg_score:.3f}")
            self.stats_labels['Coverage'].config(text=f"{summary.get('coverage', 0.0):.3f}")
            
            # Update progress bar (based on number of clicks)
            max_clicks = self.config.get('max_clicks', 100)
            progress = min(100, (summary['total_clicks'] / max_clicks) * 100)
            self.progress_var.set(progress)
            
            self.progress_label.config(text=f"{summary['total_clicks']}/{max_clicks} clicks")
            
        except Exception as e:
            print(f"Failed to update statistics: {e}")
    
    def _get_ai_suggestions(self, event=None):
        """Get AI suggestions and display them"""
        try:
            strategy = self.strategy_var.get() if hasattr(self, 'strategy_var') else self.current_strategy
            num_suggestions = self.suggestions_var.get() if hasattr(self, 'suggestions_var') else 5
            
            suggestions = self.navigator.suggest_region(strategy, num_suggestions)
            self.suggestion_points = suggestions
            
            # Update plot
            self.ui_queue.put(lambda: self._update_exploration_plot())
            
            print(f"Generated {len(suggestions)} suggestions using {strategy} strategy")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get suggestions: {e}")
    
    def _toggle_strategy(self, event=None):
        """Toggle exploration strategy"""
        strategies = ['uncertainty', 'expected_improvement', 'random']
        current_idx = strategies.index(self.current_strategy)
        self.current_strategy = strategies[(current_idx + 1) % len(strategies)]
        
        self.btn_strategy.label.set_text(f'Strategy: {self.current_strategy}')
        print(f"Switched to {self.current_strategy} strategy")
    
    def _clear_exploration(self, event=None):
        """Clear exploration history"""
        if messagebox.askyesno("Confirm", "Clear all exploration history?"):
            self.navigator.visited_positions = []
            self.navigator.scores_history = []
            self.navigator.metrics = {
                'best_score': 0.0,
                'total_clicks': 0,
                'novelty_scores': [],
                'cognitive_gains': []
            }
            
            self.suggestion_points = []
            
            # Clear UI
            self.clicks_text.delete(1.0, tk.END)
            self.ui_queue.put(lambda: self._update_exploration_plot())
    
    def _reset_exploration(self):
        """Reset exploration (same as clear for now)"""
        self._clear_exploration()
    
    def _auto_explore(self):
        """Automatically explore using AI suggestions"""
        try:
            num_clicks = 10  # Number of automatic clicks
            
            for _ in range(num_clicks):
                # Get suggestions
                suggestions = self.navigator.suggest_region(
                    self.strategy_var.get(), 1
                )
                
                if suggestions:
                    # Click first suggestion
                    position = suggestions[0]
                    self._process_click(position)
                    
                    # Brief pause for UI updates
                    threading.Event().wait(0.5)
            
            messagebox.showinfo("Complete", f"Completed {num_clicks} automatic explorations")
            
        except Exception as e:
            messagebox.showerror("Error", f"Auto exploration failed: {e}")
    
    def run(self):
        """Run the interactive UI"""
        try:
            # Show matplotlib window
            plt.show(block=False)
            
            # Start tkinter main loop
            self.root.mainloop()
            
        except KeyboardInterrupt:
            print("UI interrupted by user")
        finally:
            self.is_running = False
            plt.close('all')
    
    def close(self):
        """Close the UI"""
        self.is_running = False
        if hasattr(self, 'root'):
            self.root.quit()
            self.root.destroy()
        plt.close('all')


class BatchNavigatorUI:
    """
    Simplified UI for batch/automated navigation experiments
    """
    
    def __init__(self, navigator: LatentNavigator):
        self.navigator = navigator
        self.results_history = []
    
    def run_batch_experiment(self, 
                           num_sessions: int = 10,
                           clicks_per_session: int = 50) -> List[Dict[str, Any]]:
        """
        Run batch navigation experiments
        
        Args:
            num_sessions: Number of navigation sessions
            clicks_per_session: Number of clicks per session
            
        Returns:
            results: List of session results
        """
        results = []
        
        for session_idx in range(num_sessions):
            print(f"Running session {session_idx + 1}/{num_sessions}")
            
            # Reset navigator
            self.navigator.visited_positions = []
            self.navigator.scores_history = []
            self.navigator.metrics = {
                'best_score': 0.0,
                'total_clicks': 0,
                'novelty_scores': [],
                'cognitive_gains': []
            }
            
            # Run session
            session_results = []
            
            for click_idx in range(clicks_per_session):
                # Get suggestions
                suggestions = self.navigator.suggest_region('uncertainty', 5)
                
                # Select position (first suggestion or random)
                if suggestions:
                    position = suggestions[0]
                else:
                    position = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
                
                # Execute click
                result = self.navigator.human_click(position)
                session_results.append(result)
            
            # Store session summary
            session_summary = {
                'session_id': session_idx,
                'clicks': session_results,
                'summary': self.navigator.get_navigation_summary()
            }
            
            results.append(session_summary)
        
        self.results_history = results
        return results
    
    def get_aggregate_results(self) -> Dict[str, Any]:
        """Get aggregated results across all sessions"""
        if not self.results_history:
            return {}
        
        # Aggregate metrics
        best_scores = [session['summary']['best_score'] for session in self.results_history]
        avg_scores = [session['summary']['average_score'] for session in self.results_history]
        total_clicks = [session['summary']['total_clicks'] for session in self.results_history]
        
        aggregate = {
            'num_sessions': len(self.results_history),
            'best_score_mean': np.mean(best_scores),
            'best_score_std': np.std(best_scores),
            'avg_score_mean': np.mean(avg_scores),
            'avg_score_std': np.std(avg_scores),
            'total_clicks_mean': np.mean(total_clicks),
            'sessions': self.results_history
        }
        
        return aggregate


def create_navigator_ui(navigator: LatentNavigator, 
                       config: Dict[str, Any],
                       batch_mode: bool = False) -> NavigatorUI:
    """Factory function to create navigator UI"""
    if batch_mode:
        return BatchNavigatorUI(navigator)
    else:
        return NavigatorUI(navigator, config)
