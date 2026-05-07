"""
Enhanced Experiment Visualizer - Beautiful display of agent actions and reflections.
No external dependencies - uses ANSI escape codes directly.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List


class ExperimentVisualizer:
    """Beautiful visualization for enhanced agent experiments."""
    
    # ANSI Color Codes
    COLORS = {
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'CYAN': '\033[96m',
        'WHITE': '\033[97m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'RESET': '\033[0m'
    }
    
    AGENT_COLORS = {
        'Agent_01': COLORS['CYAN'],
        'Agent_02': COLORS['MAGENTA'], 
        'Agent_03': COLORS['YELLOW'],
        'Agent_04': COLORS['GREEN'],
        'Agent_05': COLORS['BLUE'],
    }
    
    def __init__(self):
        self.round_actions = []
    
    def get_agent_color(self, agent_id: str) -> str:
        """Get color for specific agent."""
        return self.AGENT_COLORS.get(agent_id, self.COLORS['WHITE'])
    
    def show_experiment_header(self, experiment_name: str, num_agents: int, max_rounds: int):
        """Show beautiful experiment header."""
        print(f"\n{self.COLORS['BOLD']}{self.COLORS['BLUE']}{'='*80}{self.COLORS['RESET']}")
        print(f"{self.COLORS['BOLD']}{self.COLORS['CYAN']}🧪 ENHANCED AGENT SOCIETY EXPERIMENT{self.COLORS['RESET']}")
        print(f"{self.COLORS['BOLD']}{self.COLORS['BLUE']}{'='*80}{self.COLORS['RESET']}")
        print(f"📋 Experiment: {self.COLORS['BOLD']}{experiment_name}{self.COLORS['RESET']}")
        print(f"🤖 Agents: {self.COLORS['BOLD']}{num_agents}{self.COLORS['RESET']}")
        print(f"🔄 Rounds: {self.COLORS['BOLD']}{max_rounds}{self.COLORS['RESET']}")
        print(f"⏰ Started: {self.COLORS['BOLD']}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{self.COLORS['RESET']}")
        print(f"{self.COLORS['BLUE']}{'='*80}{self.COLORS['RESET']}\n")
    
    def show_round_header(self, round_num: int, max_rounds: int):
        """Show round transition with beautiful formatting."""
        print(f"\n{self.COLORS['BOLD']}{self.COLORS['GREEN']}🔄 ROUND {round_num}/{max_rounds}{self.COLORS['RESET']}")
        print(f"{self.COLORS['GREEN']}{'─'*50}{self.COLORS['RESET']}")
    
    def show_phase_header(self, phase_name: str, icon: str):
        """Show phase header."""
        print(f"\n{self.COLORS['BOLD']}{icon} {phase_name}{self.COLORS['RESET']}")
        print(f"{self.COLORS['WHITE']}{'·'*30}{self.COLORS['RESET']}")
    
    def show_agent_reflection(self, agent_id: str, reflection: str, observation: Dict[str, Any]):
        """Show agent reflection in beautiful format."""
        color = self.get_agent_color(agent_id)
        
        print(f"\n{color}💭 {self.COLORS['BOLD']}{agent_id} REFLECTION{self.COLORS['RESET']}")
        print(f"{color}{'─'*40}{self.COLORS['RESET']}")
        
        # Show observation summary
        tools_count = len(observation.get('all_visible_tools', {}))
        neighbor_count = sum(len(tools) for tools in observation.get('neighbor_tools', {}).values())
        my_tools_count = len(observation.get('my_tools', []))
        
        print(f"   🔍 Observed: {self.COLORS['BOLD']}{tools_count}{self.COLORS['RESET']} total tools")
        print(f"   🤝 Neighbors: {self.COLORS['BOLD']}{neighbor_count}{self.COLORS['RESET']} neighbor tools")
        print(f"   🔧 My tools: {self.COLORS['BOLD']}{my_tools_count}{self.COLORS['RESET']} built")
        
        # Show reflection content (wrapped)
        print(f"\n{color}   💬 Reflection:{self.COLORS['RESET']}")
        wrapped_reflection = self._wrap_text(reflection, 70)
        for line in wrapped_reflection:
            print(f"   {color}│{self.COLORS['RESET']} {line}")
        print(f"   {color}└{'─'*70}{self.COLORS['RESET']}")
    
    def show_tool_creation(self, agent_id: str, tool_info: Dict[str, Any], success: bool):
        """Show tool creation with code preview."""
        color = self.get_agent_color(agent_id)
        
        if success:
            icon = "✅"
            status_color = self.COLORS['GREEN']
            status = "SUCCESS"
        else:
            icon = "❌"
            status_color = self.COLORS['RED']
            status = "FAILED"
        
        tool_name = tool_info.get('tool_name', 'unknown')
        
        print(f"\n{color}🔨 {self.COLORS['BOLD']}{agent_id} TOOL BUILDING{self.COLORS['RESET']}")
        print(f"{color}{'─'*40}{self.COLORS['RESET']}")
        print(f"   {icon} Tool: {self.COLORS['BOLD']}{tool_name}{self.COLORS['RESET']}")
        print(f"   📊 Status: {status_color}{self.COLORS['BOLD']}{status}{self.COLORS['RESET']}")
        
        if success and 'tool_file' in tool_info:
            # Show code preview
            tool_file = tool_info['tool_file']
            if os.path.exists(tool_file):
                print(f"\n{color}   📝 Code Preview:{self.COLORS['RESET']}")
                try:
                    with open(tool_file, 'r') as f:
                        lines = f.readlines()[:10]  # Show first 10 lines
                    
                    for i, line in enumerate(lines, 1):
                        line = line.rstrip()
                        if line:
                            print(f"   {color}│{self.COLORS['RESET']} {i:2d}: {line}")
                    
                    if len(lines) >= 10:
                        print(f"   {color}│{self.COLORS['RESET']}     ... (truncated)")
                    
                    print(f"   {color}└{'─'*70}{self.COLORS['RESET']}")
                except:
                    print(f"   {color}│{self.COLORS['RESET']} [Code preview unavailable]")
    
    def show_test_execution(self, agent_id: str, tool_name: str, test_results: Dict[str, Any]):
        """Show test execution results."""
        color = self.get_agent_color(agent_id)
        
        execution_success = test_results.get('execution_success', False)
        all_passed = test_results.get('all_passed', False)
        total_tests = test_results.get('total_tests', 0)
        passed_tests = test_results.get('passed_tests', 0)
        failed_tests = test_results.get('failed_tests', 0)
        
        # FIX: Prioritize test pass/fail status over execution status for a clearer report.
        if all_passed:
            icon = "✅"
            status_color = self.COLORS['GREEN']
            status = "ALL PASSED"
        elif passed_tests > 0 and failed_tests > 0:
            icon = "⚠️"
            status_color = self.COLORS['YELLOW']
            status = "SOME FAILED"
        elif failed_tests > 0 and passed_tests == 0:
            icon = "❌"
            status_color = self.COLORS['RED']
            status = "ALL FAILED"
        elif not execution_success:
            icon = "🔥"
            status_color = self.COLORS['RED']
            status = "EXECUTION ERROR"
        else:
            # Catch-all for unusual cases, like no tests found but execution was ok
            icon = "❓"
            status_color = self.COLORS['WHITE']
            status = "UNKNOWN STATUS"

        print(f"\n{color}🧪 {self.COLORS['BOLD']}{agent_id} TESTING{self.COLORS['RESET']}")
        print(f"{color}{'─'*40}{self.COLORS['RESET']}")
        print(f"   {icon} Tool: {self.COLORS['BOLD']}{tool_name}{self.COLORS['RESET']}")
        print(f"   📊 Status: {status_color}{self.COLORS['BOLD']}{status}{self.COLORS['RESET']}")
        print(f"   📈 Results: {self.COLORS['GREEN']}{passed_tests}✅{self.COLORS['RESET']} / {self.COLORS['RED']}{failed_tests}❌{self.COLORS['RESET']} / {total_tests} total")
        
        # Show individual test results if available
        tests = test_results.get('tests', [])
        if tests and len(tests) <= 5:  # Show details for small test suites
            print(f"   📋 Test Details:")
            for test in tests:
                test_name = test.get('name', 'unknown')
                test_passed = test.get('passed', False)
                test_icon = "✅" if test_passed else "❌"
                print(f"      {test_icon} {test_name}")
    
    def show_round_summary(self, round_num: int, round_results: Dict[str, Any]):
        """Show beautiful round summary."""
        tools_created = round_results.get('tools_created', 0)
        tests_created = round_results.get('tests_created', 0)
        tests_passed = round_results.get('tests_passed', 0)
        tests_failed = round_results.get('tests_failed', 0)
        collaboration_events = round_results.get('collaboration_events', 0)
        total_tools = round_results.get('total_tools_in_system', 0)
        
        print(f"\n{self.COLORS['BOLD']}{self.COLORS['BLUE']}📊 ROUND {round_num} SUMMARY{self.COLORS['RESET']}")
        print(f"{self.COLORS['BLUE']}{'='*50}{self.COLORS['RESET']}")
        print(f"   🔧 Tools Created: {self.COLORS['BOLD']}{tools_created}{self.COLORS['RESET']}")
        print(f"   🧪 Tests Created: {self.COLORS['BOLD']}{tests_created}{self.COLORS['RESET']}")
        print(f"   ✅ Tests Passed: {self.COLORS['GREEN']}{tests_passed}{self.COLORS['RESET']}")
        print(f"   ❌ Tests Failed: {self.COLORS['RED']}{tests_failed}{self.COLORS['RESET']}")
        print(f"   🤝 Collaborations: {self.COLORS['BOLD']}{collaboration_events}{self.COLORS['RESET']}")
        print(f"   📈 Total Tools: {self.COLORS['BOLD']}{total_tools}{self.COLORS['RESET']}")
        print(f"{self.COLORS['BLUE']}{'='*50}{self.COLORS['RESET']}")
    
    def show_experiment_summary(self, final_stats: Dict[str, Any], experiment_dir: str):
        """Show final experiment summary."""
        print(f"\n{self.COLORS['BOLD']}{self.COLORS['CYAN']}🎉 EXPERIMENT COMPLETE!{self.COLORS['RESET']}")
        print(f"{self.COLORS['CYAN']}{'='*60}{self.COLORS['RESET']}")
        
        print(f"📊 {self.COLORS['BOLD']}FINAL STATISTICS:{self.COLORS['RESET']}")
        print(f"   🔧 Total Tools: {self.COLORS['BOLD']}{final_stats.get('total_tools_created', 0)}{self.COLORS['RESET']}")
        print(f"   🧪 Total Tests: {self.COLORS['BOLD']}{final_stats.get('total_tests_created', 0)}{self.COLORS['RESET']}")
        print(f"   ✅ Test Pass Rate: {self.COLORS['GREEN']}{final_stats.get('test_pass_rate', 0):.1%}{self.COLORS['RESET']}")
        print(f"   📋 Test Coverage: {self.COLORS['BLUE']}{final_stats.get('testing_coverage_rate', 0):.1%}{self.COLORS['RESET']}")
        print(f"   🤝 Collaborations: {self.COLORS['BOLD']}{final_stats.get('total_collaboration_events', 0)}{self.COLORS['RESET']}")
        
        print(f"\n🤖 {self.COLORS['BOLD']}AGENT PRODUCTIVITY:{self.COLORS['RESET']}")
        agent_productivity = final_stats.get('agent_productivity', {})
        for agent_id, stats in agent_productivity.items():
            color = self.get_agent_color(agent_id)
            tools = stats.get('tools_built', 0)
            tests = stats.get('tests_built', 0)
            print(f"   {color}{agent_id}{self.COLORS['RESET']}: {tools} tools, {tests} tests")
        
        print(f"\n📁 {self.COLORS['BOLD']}RESULTS SAVED TO:{self.COLORS['RESET']}")
        print(f"   📂 {experiment_dir}")
        print(f"   📄 results.json, summary.txt, reflection histories")
        
        print(f"\n{self.COLORS['CYAN']}{'='*60}{self.COLORS['RESET']}")
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

    def save_visualization_log(self, experiment_dir: str):
        """Save visualization log for later review."""
        log_file = os.path.join(experiment_dir, "visualization_log.json")
        
        log_data = {
            "experiment_visualized": True,
            "total_round_actions": len(self.round_actions),
            "visualization_timestamp": datetime.now().isoformat(),
            "round_actions": self.round_actions
        }
        
        try:
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"🎨 Visualization log saved: {log_file}")
        except Exception as e:
            print(f"⚠️  Error saving visualization log: {e}")
