#!/usr/bin/env python3
"""
TCI-Lite v4: Tool Complexity Index Analyzer
==========================================

Implements a simplified 10-point Tool Complexity Index framework:
1. Code Complexity (0-3): Linear scaling based on LOC up to 300 lines
2. Interface Complexity (0-2): Parameter count + return structure complexity
3. Compositional Complexity (0-5): Tool calls (4pts) + external imports (1pt)

Usage:
    python complexity_analyzer.py --analyze shared_tools_template
    python complexity_analyzer.py --experiment experiments/enhanced_testing_demo_20250902_082953
"""

import os
import ast
import json
import argparse
import re
from typing import Dict, List, Any, Set, Optional
from pathlib import Path


class TCILiteAnalyzer:
    """TCI-Lite v4: Simplified 10-point scale complexity analyzer"""
    
    def __init__(self):
        # Standard library modules to exclude from external import counting
        self.stdlib_modules = {
            'os', 'sys', 'json', 'math', 're', 'datetime', 'time', 'random',
            'collections', 'itertools', 'functools', 'pathlib', 'typing',
            'ast', 'inspect', 'importlib', 'statistics', 'argparse', 'logging',
            'urllib', 'http', 'email', 'html', 'xml', 'csv', 'configparser',
            'tempfile', 'shutil', 'glob', 'fnmatch', 'linecache', 'pickle',
            'copy', 'pprint', 'textwrap', 'string', 'io', 'contextlib'
        }
    
    def analyze_tools_directory(self, tools_dir: str) -> Dict[str, Dict[str, Any]]:
        """Analyze all tools in a directory and return TCI metrics"""
        tools_path = Path(tools_dir)
        if not tools_path.exists():
            raise FileNotFoundError(f"Tools directory not found: {tools_dir}")
        
        # Find all Python tool files
        tool_files = [f for f in tools_path.glob("*.py") if f.stem != '__init__']
        all_tools = {f.stem for f in tool_files}
        
        results = {}
        for tool_file in tool_files:
            tool_name = tool_file.stem
            metrics = self.analyze_single_tool(tool_file, all_tools)
            results[tool_name] = metrics
            
        return results
    
    def analyze_single_tool(self, tool_file: Path, all_tools: Set[str]) -> Dict[str, Any]:
        """Analyze a single tool file and return TCI metrics"""
        try:
            # Calculate each complexity dimension
            code_score = self._calculate_code_complexity(tool_file)
            iface_score = self._calculate_interface_complexity(tool_file)
            comp_score = self._calculate_compositional_complexity(tool_file, all_tools)
            
            # Quality gate (simplified - just check if file is valid Python)
            quality_gate = 1.0 if self._is_valid_python(tool_file) else 0.6
            
            # Final TCI score
            tci_raw = code_score + iface_score + comp_score
            tci_final = quality_gate * tci_raw
            
            return {
                'tci_score': round(tci_final, 3),
                'tci_raw': round(tci_raw, 3),
                'code_complexity': round(code_score, 3),
                'interface_complexity': round(iface_score, 3),
                'compositional_complexity': round(comp_score, 3),
                'quality_gate': quality_gate,
                # Detailed metrics for analysis
                'lines_of_code': self._count_loc(tool_file),
                'param_count': self._count_parameters(tool_file),
                'tool_calls': self._count_unique_tool_calls(tool_file, all_tools),
                'external_imports': self._count_external_imports(tool_file)
            }
            
        except Exception as e:
            print(f"Error analyzing {tool_file}: {e}")
            return {
                'tci_score': 0.0,
                'tci_raw': 0.0,
                'code_complexity': 0.0,
                'interface_complexity': 0.0,
                'compositional_complexity': 0.0,
                'quality_gate': 0.0
            }
    
    def _calculate_code_complexity(self, tool_file: Path) -> float:
        """Code complexity: Linear scaling to 300 lines, max 3 points"""
        loc = self._count_loc(tool_file)
        return 3.0 * min(1.0, loc / 300.0)
    
    def _calculate_interface_complexity(self, tool_file: Path) -> float:
        """Interface complexity: Parameter count + return complexity, max 2 points"""
        try:
            with open(tool_file, 'r') as f:
                code = f.read()
            tree = ast.parse(code)
            
            # Find execute function
            execute_func = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'execute':
                    execute_func = node
                    break
            
            if not execute_func:
                return 0.0
            
            # Parameter complexity (0-1 points)
            param_count = len(execute_func.args.args)
            param_score = min(1.0, param_count / 5.0)
            
            # Return complexity (0-1 points)
            return_score = self._analyze_return_complexity(execute_func)
            
            return param_score + return_score
            
        except Exception:
            return 0.0
    
    def _calculate_compositional_complexity(self, tool_file: Path, all_tools: Set[str]) -> float:
        """Compositional complexity: Tool calls (4pts) + external imports (1pt), max 5 points"""
        # Tool calls complexity (0-4 points, max 8 different tools)
        tool_calls = self._count_unique_tool_calls(tool_file, all_tools)
        call_score = min(4.0, tool_calls * 0.5)
        
        # External imports complexity (0-1 points, max 10 libraries)
        imports = self._count_external_imports(tool_file)
        import_score = min(1.0, imports * 0.1)
        
        return call_score + import_score
    
    def _count_loc(self, tool_file: Path) -> int:
        """Count lines of code excluding comments and empty lines"""
        try:
            with open(tool_file, 'r') as f:
                lines = f.readlines()
            
            loc = 0
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    loc += 1
            return loc
            
        except Exception:
            return 0
    
    def _count_parameters(self, tool_file: Path) -> int:
        """Count parameters in execute function"""
        try:
            with open(tool_file, 'r') as f:
                code = f.read()
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'execute':
                    return len(node.args.args)
            return 0
            
        except Exception:
            return 0
    
    def _analyze_return_complexity(self, func_node: ast.FunctionDef) -> float:
        """Analyze return value complexity from AST"""
        return_statements = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value:
                return_statements.append(node.value)
        
        if not return_statements:
            return 0.0
        
        # Analyze the most complex return statement
        max_complexity = 0.0
        for ret in return_statements:
            if isinstance(ret, ast.Dict):
                # Dictionary return - count keys
                complexity = min(1.0, len(ret.keys) / 5.0)
            elif isinstance(ret, (ast.List, ast.Tuple)):
                # List/tuple return - count elements
                complexity = min(1.0, len(ret.elts) / 10.0) 
            elif isinstance(ret, ast.Call):
                # Function call return - moderate complexity
                complexity = 0.5
            else:
                # Simple return value
                complexity = 0.3
            
            max_complexity = max(max_complexity, complexity)
        
        return max_complexity
    
    def _count_unique_tool_calls(self, tool_file: Path, all_tools: Set[str]) -> int:
        """Count unique tool calls in the file"""
        try:
            with open(tool_file, 'r') as f:
                code = f.read()
            
            # Pattern to match context.call_tool("tool_name")
            pattern = r'context\.call_tool\(["\']([^"\']+)["\']\s*[,)]'
            matches = re.findall(pattern, code)
            
            # Only count tools that actually exist
            unique_tools = set()
            for tool_name in matches:
                if tool_name in all_tools:
                    unique_tools.add(tool_name)
            
            return len(unique_tools)
            
        except Exception:
            return 0
    
    def _count_external_imports(self, tool_file: Path) -> int:
        """Count external library imports (excluding standard library)"""
        try:
            with open(tool_file, 'r') as f:
                code = f.read()
            
            import_count = 0
            for line in code.split('\n'):
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    module_name = self._extract_module_name(line)
                    if module_name and module_name not in self.stdlib_modules:
                        import_count += 1
            
            return import_count
            
        except Exception:
            return 0
    
    def _extract_module_name(self, import_line: str) -> Optional[str]:
        """Extract the base module name from an import statement"""
        try:
            if import_line.startswith('import '):
                # import pandas.core -> pandas
                module = import_line.split()[1].split('.')[0]
            elif import_line.startswith('from '):
                # from sklearn.metrics import -> sklearn
                module = import_line.split()[1].split('.')[0]
            else:
                return None
            
            return module.strip()
            
        except Exception:
            return None
    
    def _is_valid_python(self, tool_file: Path) -> bool:
        """Check if file is valid Python and has execute function"""
        try:
            with open(tool_file, 'r') as f:
                code = f.read()
            tree = ast.parse(code)
            
            # Check if execute function exists
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'execute':
                    return True
            return False
            
        except Exception:
            return False


class TCIAnalyzer:
    """Main TCI analyzer that orchestrates complexity analysis"""
    
    def __init__(self):
        self.tci_analyzer = TCILiteAnalyzer()
    
    def analyze_tools_in_directory(self, tools_dir: str) -> Dict[str, Dict[str, Any]]:
        """Analyze all tools in a directory"""
        return self.tci_analyzer.analyze_tools_directory(tools_dir)
    
    def analyze_experiment_directory(self, exp_dir: str) -> Dict[str, Any]:
        """Analyze all agent directories in an experiment"""
        exp_path = Path(exp_dir)
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        
        results = {}
        personal_tools_dir = exp_path / "personal_tools"
        
        if personal_tools_dir.exists():
            for agent_dir in personal_tools_dir.iterdir():
                if agent_dir.is_dir() and agent_dir.name.startswith(('Agent_', 'Boid_')):
                    agent_name = agent_dir.name
                    try:
                        agent_results = self.analyze_tools_in_directory(str(agent_dir))
                        results[agent_name] = agent_results
                    except Exception as e:
                        print(f"Error analyzing {agent_name}: {e}")
                        results[agent_name] = {}
        
        return results
    
    def compute_summary_statistics(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics across all analyzed tools"""
        all_scores = []
        code_scores = []
        iface_scores = []
        comp_scores = []
        
        for agent_name, agent_tools in results.items():
            for tool_name, metrics in agent_tools.items():
                if isinstance(metrics, dict) and 'tci_score' in metrics:
                    all_scores.append(metrics['tci_score'])
                    code_scores.append(metrics.get('code_complexity', 0))
                    iface_scores.append(metrics.get('interface_complexity', 0))
                    comp_scores.append(metrics.get('compositional_complexity', 0))
        
        if not all_scores:
            return {'error': 'No valid TCI scores found'}

        return {
            'total_tools': len(all_scores),
            'tci_mean': round(sum(all_scores) / len(all_scores), 3),
            'tci_max': round(max(all_scores), 3),
            'tci_min': round(min(all_scores), 3),
            'code_mean': round(sum(code_scores) / len(code_scores), 3),
            'interface_mean': round(sum(iface_scores) / len(iface_scores), 3),
            'compositional_mean': round(sum(comp_scores) / len(comp_scores), 3),
            'high_complexity_tools': len([s for s in all_scores if s >= 7.0]),
            'medium_complexity_tools': len([s for s in all_scores if 3.0 <= s < 7.0]),
            'low_complexity_tools': len([s for s in all_scores if s < 3.0])
        }


def main():
    """Command line interface for TCI analysis"""
    parser = argparse.ArgumentParser(description='TCI-Lite v4 Complexity Analyzer')
    parser.add_argument('--analyze', type=str, help='Analyze tools in directory')
    parser.add_argument('--experiment', type=str, help='Analyze experiment directory')
    parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    analyzer = TCIAnalyzer()
    
    if args.analyze:
        print(f"Analyzing tools in: {args.analyze}")
        results = analyzer.analyze_tools_in_directory(args.analyze)
        summary = analyzer.compute_summary_statistics({'main': results})
        
        print("\n=== TCI Analysis Results ===")
        for tool_name, metrics in results.items():
            if isinstance(metrics, dict):
                print(f"{tool_name}: TCI={metrics.get('tci_score', 0):.2f} "
                      f"(Code={metrics.get('code_complexity', 0):.1f}, "
                      f"Interface={metrics.get('interface_complexity', 0):.1f}, "
                      f"Comp={metrics.get('compositional_complexity', 0):.1f})")
        
        print(f"\nSummary: {summary['total_tools']} tools, "
              f"Mean TCI={summary['tci_mean']:.2f}, "
              f"Range=[{summary['tci_min']:.2f}, {summary['tci_max']:.2f}]")
    
    elif args.experiment:
        print(f"Analyzing experiment: {args.experiment}")
        results = analyzer.analyze_experiment_directory(args.experiment)
        summary = analyzer.compute_summary_statistics(results)
        
        print("\n=== Experiment TCI Analysis ===")
        for agent_name, agent_tools in results.items():
            if agent_tools:
                agent_scores = [m.get('tci_score', 0) for m in agent_tools.values() if isinstance(m, dict)]
                if agent_scores:
                    print(f"{agent_name}: {len(agent_scores)} tools, "
                          f"Mean TCI={sum(agent_scores)/len(agent_scores):.2f}")
        
        print(f"\nOverall Summary: {summary}")
    
    else:
        parser.print_help()
    
    # Save results if output specified
    if args.output and 'results' in locals():
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main() 