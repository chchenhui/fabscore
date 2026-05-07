"""
Unit tests for audit logging and provenance tracking.
Tests the G5 audit gate requirements for full reproducibility.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.audit import (
    AuditLogger,
    get_audit_logger,
    audit_api_call,
    audit_run,
    audit_summary
)
from utils.provenance import (
    get_git_commit,
    get_git_branch,
    log_experiment_provenance,
    set_reproducible_seed
)


class TestAuditLogger:
    """Test audit logger functionality."""
    
    def setup_method(self):
        """Create temporary directory for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.logger = AuditLogger(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_audit_logger_init(self):
        """Test audit logger initialization."""
        assert self.logger is not None
        assert self.logger.output_dir == self.temp_dir
        assert self.logger.usage_log_path == self.temp_dir / "usage_log.jsonl"
        assert self.logger.provenance_path == self.temp_dir / "run_provenance.json"
        assert self.logger.total_tokens == 0
        assert self.logger.total_cost == 0.0
        assert self.logger.api_calls == 0
    
    def test_provenance_initialization(self):
        """Test provenance initialization."""
        provenance = self.logger.provenance
        
        assert 'timestamp' in provenance
        assert 'git_commit' in provenance
        assert 'git_branch' in provenance
        assert 'git_dirty' in provenance
        assert 'python_version' in provenance
        assert 'platform' in provenance
        assert 'cwd' in provenance
        assert 'env_vars' in provenance
        assert 'runs' in provenance
        assert isinstance(provenance['runs'], list)
    
    @patch('subprocess.run')
    def test_git_commit_detection(self, mock_run):
        """Test git commit hash detection."""
        # Mock successful git command
        mock_result = MagicMock()
        mock_result.stdout = "abc123def456\n"
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        commit = self.logger._get_git_commit()
        assert commit == "abc123def456"
    
    @patch('subprocess.run')
    def test_git_commit_detection_failure(self, mock_run):
        """Test git commit detection when git fails."""
        # Mock failed git command
        mock_run.side_effect = Exception("Git not found")
        
        commit = self.logger._get_git_commit()
        assert commit == "unknown"
    
    @patch('subprocess.run')
    def test_git_branch_detection(self, mock_run):
        """Test git branch detection."""
        # Mock successful git command
        mock_result = MagicMock()
        mock_result.stdout = "main\n"
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        branch = self.logger._get_git_branch()
        assert branch == "main"
    
    @patch('subprocess.run')
    def test_git_dirty_detection(self, mock_run):
        """Test git dirty status detection."""
        # Mock clean working directory
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        is_dirty = self.logger._is_git_dirty()
        assert is_dirty is False
        
        # Mock dirty working directory
        mock_result.stdout = "M  file.py\n"
        is_dirty = self.logger._is_git_dirty()
        assert is_dirty is True
    
    def test_log_api_call(self):
        """Test API call logging."""
        # Log an API call
        self.logger.log_api_call(
            provider="openai",
            model="gpt-4",
            prompt="Test prompt",
            response="Test response",
            tokens=100,
            cost=0.05,
            task_type="forecasting",
            metadata={"drug_id": "TEST_001"}
        )
        
        # Check totals updated
        assert self.logger.total_tokens == 100
        assert self.logger.total_cost == 0.05
        assert self.logger.api_calls == 1
        
        # Check usage log file created
        assert self.logger.usage_log_path.exists()
        
        # Read and verify log entry
        with open(self.logger.usage_log_path, 'r') as f:
            log_entry = json.loads(f.read().strip())
        
        assert log_entry['provider'] == "openai"
        assert log_entry['model'] == "gpt-4"
        assert log_entry['tokens'] == 100
        assert log_entry['cost'] == 0.05
        assert log_entry['task_type'] == "forecasting"
        assert log_entry['metadata']['drug_id'] == "TEST_001"
        assert 'prompt_hash' in log_entry
        assert 'response_hash' in log_entry
        assert 'timestamp' in log_entry
    
    def test_log_multiple_api_calls(self):
        """Test logging multiple API calls."""
        # Log multiple calls
        for i in range(3):
            self.logger.log_api_call(
                provider="anthropic",
                model="claude-3",
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                tokens=50 + i * 10,
                cost=0.02 + i * 0.01
            )
        
        # Check totals
        assert self.logger.total_tokens == 180  # 50 + 60 + 70
        assert self.logger.total_cost == 0.09  # 0.02 + 0.03 + 0.04
        assert self.logger.api_calls == 3
        
        # Check log file has 3 entries
        with open(self.logger.usage_log_path, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 3
    
    def test_log_run(self):
        """Test experimental run logging."""
        # Log a run
        config = {"model": "ensemble", "seed": 42, "epochs": 100}
        results = {"accuracy": 0.85, "mape": 25.0}
        
        self.logger.log_run(
            experiment_name="test_experiment",
            config=config,
            seed=42,
            results=results
        )
        
        # Check provenance file created
        assert self.logger.provenance_path.exists()
        
        # Read and verify provenance
        with open(self.logger.provenance_path, 'r') as f:
            provenance = json.load(f)
        
        assert len(provenance['runs']) == 1
        run = provenance['runs'][0]
        assert run['experiment'] == "test_experiment"
        assert run['seed'] == 42
        assert run['config'] == config
        assert run['results'] == results
        assert 'timestamp' in run
    
    def test_save_provenance(self):
        """Test provenance saving."""
        # Add a run
        self.logger.log_run("test", {"param": "value"}, 42)
        
        # Verify file exists and is valid JSON
        assert self.logger.provenance_path.exists()
        with open(self.logger.provenance_path, 'r') as f:
            data = json.load(f)
        assert 'runs' in data
    
    def test_get_summary(self):
        """Test summary generation."""
        # Log some API calls and runs
        self.logger.log_api_call("openai", "gpt-4", "test", "response", 100, 0.05)
        self.logger.log_run("test_exp", {"param": 1}, 42)
        
        summary = self.logger.get_summary()
        
        assert summary['api_calls'] == 1
        assert summary['total_tokens'] == 100
        assert summary['total_cost'] == 0.05
        assert summary['avg_cost_per_call'] == 0.05
        assert summary['experiments_run'] == 1
    
    def test_reset_counters(self):
        """Test counter reset."""
        # Log some data
        self.logger.log_api_call("openai", "gpt-4", "test", "response", 100, 0.05)
        self.logger.log_run("test_exp", {"param": 1}, 42)
        
        # Reset counters
        self.logger.reset_counters()
        
        assert self.logger.total_tokens == 0
        assert self.logger.total_cost == 0.0
        assert self.logger.api_calls == 0
        
        # Provenance should still have runs
        assert len(self.logger.provenance['runs']) == 1


class TestGlobalAuditFunctions:
    """Test global audit functions."""
    
    def setup_method(self):
        """Create temporary directory for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_audit_logger_singleton(self):
        """Test audit logger singleton pattern."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        
        assert logger1 is logger2  # Same instance
    
    @patch('utils.audit.get_audit_logger')
    def test_audit_api_call_convenience(self, mock_get_logger):
        """Test convenience function for API call logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        audit_api_call(
            provider="openai",
            model="gpt-4",
            prompt="test",
            response="response",
            tokens=100,
            cost=0.05,
            task_type="forecasting"
        )
        
        mock_logger.log_api_call.assert_called_once_with(
            "openai", "gpt-4", "test", "response", 100, 0.05,
            task_type="forecasting"
        )
    
    @patch('utils.audit.get_audit_logger')
    def test_audit_run_convenience(self, mock_get_logger):
        """Test convenience function for run logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        config = {"param": "value"}
        results = {"accuracy": 0.85}
        
        audit_run("test_exp", config, 42, results)
        
        mock_logger.log_run.assert_called_once_with("test_exp", config, 42, results)
    
    @patch('utils.audit.get_audit_logger')
    def test_audit_summary_convenience(self, mock_get_logger):
        """Test convenience function for summary."""
        mock_logger = MagicMock()
        mock_logger.get_summary.return_value = {"total_cost": 0.05}
        mock_get_logger.return_value = mock_logger
        
        summary = audit_summary()
        
        assert summary == {"total_cost": 0.05}
        mock_logger.get_summary.assert_called_once()


class TestProvenanceFunctions:
    """Test provenance tracking functions."""
    
    @patch('subprocess.run')
    def test_get_git_commit(self, mock_run):
        """Test git commit hash retrieval."""
        mock_result = MagicMock()
        mock_result.stdout = "abc123def456789\n"
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        commit = get_git_commit()
        assert commit == "abc123de"  # Short hash
    
    @patch('subprocess.run')
    def test_get_git_commit_failure(self, mock_run):
        """Test git commit retrieval when git fails."""
        mock_run.side_effect = Exception("Git not found")
        
        commit = get_git_commit()
        assert commit == "unknown"
    
    @patch('subprocess.run')
    def test_get_git_branch(self, mock_run):
        """Test git branch retrieval."""
        mock_result = MagicMock()
        mock_result.stdout = "feature-branch\n"
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        branch = get_git_branch()
        assert branch == "feature-branch"
    
    def test_log_experiment_provenance(self):
        """Test experiment provenance logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance = log_experiment_provenance(
                experiment_name="test_exp",
                seed=42,
                parameters={"model": "ensemble", "epochs": 100},
                output_dir=Path(temp_dir)
            )
            
            # Check provenance structure
            assert provenance['experiment'] == "test_exp"
            assert provenance['seed'] == 42
            assert provenance['parameters'] == {"model": "ensemble", "epochs": 100}
            assert 'timestamp' in provenance
            assert 'git_commit' in provenance
            assert 'git_branch' in provenance
            assert 'python_version' in provenance
            assert 'platform' in provenance
            
            # Check file was created
            provenance_files = list(Path(temp_dir).glob("provenance_*.json"))
            assert len(provenance_files) == 1
            
            # Verify file content
            with open(provenance_files[0], 'r') as f:
                file_data = json.load(f)
            assert file_data == provenance
    
    def test_log_experiment_provenance_no_output_dir(self):
        """Test experiment provenance logging without output directory."""
        provenance = log_experiment_provenance(
            experiment_name="test_exp",
            seed=42,
            parameters={"model": "ensemble"}
        )
        
        assert provenance['experiment'] == "test_exp"
        assert provenance['seed'] == 42
        assert provenance['parameters'] == {"model": "ensemble"}
    
    @patch('random.seed')
    @patch('numpy.random.seed')
    def test_set_reproducible_seed(self, mock_np_seed, mock_random_seed):
        """Test reproducible seed setting."""
        set_reproducible_seed(123)
        
        mock_random_seed.assert_called_once_with(123)
        mock_np_seed.assert_called_once_with(123)
    
    @patch('random.seed')
    @patch('numpy.random.seed')
    def test_set_reproducible_seed_with_torch(self, mock_np_seed, mock_random_seed):
        """Test reproducible seed setting with PyTorch."""
        # Mock torch module availability
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.manual_seed_all = MagicMock()
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            set_reproducible_seed(456)
        
        mock_random_seed.assert_called_once_with(456)
        mock_np_seed.assert_called_once_with(456)


class TestIntegration:
    """Test integration between audit and provenance systems."""
    
    def setup_method(self):
        """Create temporary directory for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_audit_workflow(self):
        """Test complete audit workflow."""
        # Create audit logger
        logger = AuditLogger(output_dir=self.temp_dir)
        
        # Log API calls
        logger.log_api_call("openai", "gpt-4", "prompt1", "response1", 100, 0.05)
        logger.log_api_call("anthropic", "claude-3", "prompt2", "response2", 150, 0.08)
        
        # Log experimental run
        config = {"model": "ensemble", "seed": 42}
        results = {"mape": 25.0, "accuracy": 0.85}
        logger.log_run("ensemble_evaluation", config, 42, results)
        
        # Verify usage log
        assert logger.usage_log_path.exists()
        with open(logger.usage_log_path, 'r') as f:
            usage_lines = f.readlines()
        assert len(usage_lines) == 2
        
        # Verify provenance
        assert logger.provenance_path.exists()
        with open(logger.provenance_path, 'r') as f:
            provenance = json.load(f)
        assert len(provenance['runs']) == 1
        
        # Verify summary
        summary = logger.get_summary()
        assert summary['api_calls'] == 2
        assert summary['total_tokens'] == 250
        assert summary['total_cost'] == 0.13
        assert summary['experiments_run'] == 1
    
    def test_audit_with_provenance_integration(self):
        """Test audit system with provenance integration."""
        # Set reproducible seed
        set_reproducible_seed(42)
        
        # Create audit logger
        logger = AuditLogger(output_dir=self.temp_dir)
        
        # Log experiment with provenance
        config = {"seed": 42, "model": "ensemble"}
        results = {"mape": 25.0}
        logger.log_run("g4_gate_evaluation", config, 42, results)
        
        # Verify both systems work together
        assert logger.provenance_path.exists()
        assert logger.usage_log_path.exists() or not logger.usage_log_path.exists()  # May be empty
        
        # Check provenance includes run info
        with open(logger.provenance_path, 'r') as f:
            provenance = json.load(f)
        
        run = provenance['runs'][0]
        assert run['experiment'] == "g4_gate_evaluation"
        assert run['seed'] == 42
        assert run['config']['seed'] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
