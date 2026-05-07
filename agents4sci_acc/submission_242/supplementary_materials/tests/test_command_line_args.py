"""
Tests for command line argument parsing, especially for the new bid cooloff feature.
"""
import pytest
from unittest.mock import patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marketplace.true_gpt_marketplace import parse_arguments, main


class TestCommandLineArguments:
    """Test command line argument parsing for marketplace simulation"""

    def test_default_bid_cooloff_rounds(self):
        """Test that bid-cooloff-rounds has correct default value"""
        with patch('sys.argv', ['script_name']):
            args = parse_arguments()
            assert args.bid_cooloff_rounds == 5

    def test_custom_bid_cooloff_rounds(self):
        """Test setting custom bid cooloff rounds"""
        test_cases = [
            (['script_name', '--bid-cooloff-rounds', '3'], 3),
            (['script_name', '--bid-cooloff-rounds', '10'], 10),
            (['script_name', '--bid-cooloff-rounds', '0'], 0),
            (['script_name', '--bid-cooloff-rounds', '1'], 1),
            (['script_name', '--bid-cooloff-rounds', '100'], 100),
        ]
        
        for argv, expected_value in test_cases:
            with patch('sys.argv', argv):
                args = parse_arguments()
                assert args.bid_cooloff_rounds == expected_value, f"Failed for argv: {argv}"

    def test_bid_cooloff_rounds_help_text(self):
        """Test that bid cooloff rounds argument has proper help text"""
        with patch('sys.argv', ['script_name', '--help']):
            with pytest.raises(SystemExit) as exc_info:  # argparse exits with help
                parse_arguments()
            # Help should have been displayed with exit code 0
            assert exc_info.value.code == 0

    def test_bid_cooloff_rounds_with_other_args(self):
        """Test bid cooloff rounds in combination with other arguments"""
        argv = [
            'script_name',
            '--freelancers', '10',
            '--clients', '5',
            '--rounds', '15',
            '--bid-cooloff-rounds', '7',
            '--bids-per-round', '2',
            '--jobs-per-freelancer', '4',
            '--job-selection', 'relevance',
            '--relevance-mode', 'moderate'
        ]
        
        with patch('sys.argv', argv):
            args = parse_arguments()
            assert args.freelancers == 10
            assert args.clients == 5
            assert args.rounds == 15
            assert args.bid_cooloff_rounds == 7
            assert args.bids_per_round == 2
            assert args.jobs_per_freelancer == 4
            assert args.job_selection == 'relevance'
            assert args.relevance_mode == 'moderate'

    def test_invalid_bid_cooloff_rounds(self):
        """Test handling of invalid bid cooloff rounds values"""
        # Test non-integer value
        with patch('sys.argv', ['script_name', '--bid-cooloff-rounds', 'invalid']):
            with pytest.raises(SystemExit):
                parse_arguments()

        # Test float value (should be converted to int or error)
        with patch('sys.argv', ['script_name', '--bid-cooloff-rounds', '5.5']):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_negative_bid_cooloff_rounds_allowed(self):
        """Test that negative values are accepted (marketplace handles them)"""
        with patch('sys.argv', ['script_name', '--bid-cooloff-rounds', '-1']):
            args = parse_arguments()
            assert args.bid_cooloff_rounds == -1

    def test_all_default_arguments(self):
        """Test that all arguments have sensible defaults"""
        with patch('sys.argv', ['script_name']):
            args = parse_arguments()
            
            # Check all defaults
            assert args.freelancers == 5
            assert args.clients == 3
            assert args.rounds == 6
            assert args.bids_per_round == 1
            assert args.jobs_per_freelancer == 3
            assert args.job_selection == 'random'
            assert args.relevance_mode == 'strict'
            assert args.reflection_probability == 0.1
            assert args.bid_cooloff_rounds == 5  # Our new parameter
            assert args.max_workers == 15
            assert args.no_cache is False
            assert args.disable_reflections is False
            assert args.scenario is None

    def test_scenario_with_cooloff(self):
        """Test that scenarios work properly with bid cooloff parameter"""
        scenarios = ['scarcity', 'moderate', 'abundant']
        
        for scenario in scenarios:
            with patch('sys.argv', ['script_name', '--scenario', scenario, '--bid-cooloff-rounds', '8']):
                args = parse_arguments()
                assert args.scenario == scenario
                assert args.bid_cooloff_rounds == 8


class TestMainFunctionIntegration:
    """Test the main function handles bid cooloff arguments correctly"""

    @patch('marketplace.true_gpt_marketplace.TrueGPTMarketplace')
    @patch('marketplace.true_gpt_marketplace.get_token_tracker')
    def test_main_passes_cooloff_to_marketplace(self, mock_token_tracker, mock_marketplace):
        """Test that main function passes bid cooloff parameter to marketplace"""
        # Mock the marketplace instance and its methods
        mock_instance = mock_marketplace.return_value
        mock_instance.run_simulation.return_value = {}
        
        # Mock token tracker
        mock_tracker = mock_token_tracker.return_value
        mock_tracker.save_report.return_value = None
        
        # Test with custom bid cooloff rounds
        with patch('sys.argv', ['script_name', '--bid-cooloff-rounds', '12']):
            try:
                main()
            except SystemExit:
                pass  # main() might exit after completion
            
            # Verify marketplace was called with correct parameters
            mock_marketplace.assert_called_once()
            call_args = mock_marketplace.call_args
            
            # Check that bid_cooloff_rounds was passed correctly
            assert 'bid_cooloff_rounds' in call_args.kwargs
            assert call_args.kwargs['bid_cooloff_rounds'] == 12

    @patch('marketplace.true_gpt_marketplace.TrueGPTMarketplace')
    @patch('marketplace.true_gpt_marketplace.get_token_tracker')
    def test_main_default_cooloff_to_marketplace(self, mock_token_tracker, mock_marketplace):
        """Test that main function uses default bid cooloff when not specified"""
        # Mock the marketplace instance and its methods
        mock_instance = mock_marketplace.return_value
        mock_instance.run_simulation.return_value = {}
        
        # Mock token tracker
        mock_tracker = mock_token_tracker.return_value
        mock_tracker.save_report.return_value = None
        
        # Test with default arguments
        with patch('sys.argv', ['script_name']):
            try:
                main()
            except SystemExit:
                pass  # main() might exit after completion
            
            # Verify marketplace was called with default cooloff
            mock_marketplace.assert_called_once()
            call_args = mock_marketplace.call_args
            
            # Check that bid_cooloff_rounds was passed with default value
            assert 'bid_cooloff_rounds' in call_args.kwargs
            assert call_args.kwargs['bid_cooloff_rounds'] == 5  # Default value

    @patch('marketplace.true_gpt_marketplace.TrueGPTMarketplace')
    @patch('marketplace.true_gpt_marketplace.get_token_tracker')
    def test_main_disabled_cooloff_to_marketplace(self, mock_token_tracker, mock_marketplace):
        """Test that main function handles disabled cooloff (0)"""
        # Mock the marketplace instance and its methods
        mock_instance = mock_marketplace.return_value
        mock_instance.run_simulation.return_value = {}
        
        # Mock token tracker
        mock_tracker = mock_token_tracker.return_value
        mock_tracker.save_report.return_value = None
        
        # Test with disabled cooloff
        with patch('sys.argv', ['script_name', '--bid-cooloff-rounds', '0']):
            try:
                main()
            except SystemExit:
                pass  # main() might exit after completion
            
            # Verify marketplace was called with disabled cooloff
            mock_marketplace.assert_called_once()
            call_args = mock_marketplace.call_args
            
            # Check that bid_cooloff_rounds was passed as 0
            assert 'bid_cooloff_rounds' in call_args.kwargs
            assert call_args.kwargs['bid_cooloff_rounds'] == 0

    def test_cooloff_logging_output(self, capsys):
        """Test that bid cooloff configuration is properly logged"""
        with patch('marketplace.true_gpt_marketplace.TrueGPTMarketplace') as mock_marketplace:
            mock_instance = mock_marketplace.return_value
            mock_instance.run_simulation.return_value = {}
            
            with patch('marketplace.true_gpt_marketplace.get_token_tracker') as mock_token_tracker:
                mock_tracker = mock_token_tracker.return_value
                mock_tracker.save_report.return_value = None
                
                # Test enabled cooloff logging
                with patch('sys.argv', ['script_name', '--bid-cooloff-rounds', '8']):
                    try:
                        main()
                    except SystemExit:
                        pass
                    
                    captured = capsys.readouterr()
                    assert "🔄 Bid Cooloff: Freelancers can re-bid after 8 rounds" in captured.out

    def test_disabled_cooloff_logging_output(self, capsys):
        """Test logging output when cooloff is disabled"""
        with patch('marketplace.true_gpt_marketplace.TrueGPTMarketplace') as mock_marketplace:
            mock_instance = mock_marketplace.return_value
            mock_instance.run_simulation.return_value = {}
            
            with patch('marketplace.true_gpt_marketplace.get_token_tracker') as mock_token_tracker:
                mock_tracker = mock_token_tracker.return_value
                mock_tracker.save_report.return_value = None
                
                # Test disabled cooloff logging
                with patch('sys.argv', ['script_name', '--bid-cooloff-rounds', '0']):
                    try:
                        main()
                    except SystemExit:
                        pass
                    
                    captured = capsys.readouterr()
                    assert "🔄 Bid Cooloff: Disabled (freelancers cannot re-bid on previously declined jobs)" in captured.out
