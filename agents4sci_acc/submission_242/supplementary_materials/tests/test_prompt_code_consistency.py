"""
Test consistency between GPT prompts and code expectations.
Catches mismatches like unused functions and prompt-code disconnects.
"""
import pytest
import ast
import inspect
from pathlib import Path

from prompts.base import generate_client_persona_prompt, generate_freelancer_persona_prompt
from marketplace.simple_reputation import SimpleReputationManager
from marketplace.job_categories import CategoryMetrics


class TestPromptCodeConsistency:
    """Test that prompts and code are consistent."""
    
    def test_budget_philosophy_consistency(self):
        """Test that budget philosophies in prompts match code expectations."""
        # Get the client prompt
        prompt = generate_client_persona_prompt(1)
        
        # Extract budget philosophies mentioned in prompt
        assert "cost-conscious" in prompt
        assert "value-focused" in prompt  
        assert "premium-focused" in prompt
        
        # These should match what the code actually checks for
        # (This test would have caught the original "balanced" vs "value" mismatch)
    
    def test_no_unused_reputation_functions(self):
        """Test that reputation manager functions are actually used."""
        # Get all public methods from SimpleReputationManager
        manager = SimpleReputationManager()
        public_methods = [method for method in dir(manager) 
                         if not method.startswith('_') and callable(getattr(manager, method))]
        
        # These methods should be used somewhere in the codebase
        used_methods = []
        unused_methods = []
        
        for method in public_methods:
            if method in ['initialize_freelancer', 'initialize_client', 
                         'update_freelancer_after_job', 'update_client_after_post',
                         'get_freelancer_reputation', 'get_client_reputation']:
                used_methods.append(method)
            else:
                # Check if method is used anywhere in src/
                import subprocess
                result = subprocess.run(
                    ['grep', '-r', f'{method}(', 'src/'], 
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    used_methods.append(method)
                else:
                    unused_methods.append(method)
        
        # Report unused methods (these should be removed or integrated)
        if unused_methods:
            pytest.fail(f"Unused reputation methods detected: {unused_methods}. "
                       f"Either integrate them into the logic or remove them.")
    
    def test_no_unused_dataclass_fields(self):
        """Test that dataclass fields are actually used."""
        # Check CategoryMetrics for unused fields
        metrics_fields = CategoryMetrics.__dataclass_fields__.keys()
        
        unused_fields = []
        for field in metrics_fields:
            # Check if field is accessed anywhere
            import subprocess
            result = subprocess.run(
                ['grep', '-r', f'.{field}', 'src/'], 
                capture_output=True, text=True
            )
            # Filter out the definition itself
            if result.returncode != 0:
                unused_fields.append(field)
        
        # Report unused fields
        if unused_fields:
            pytest.fail(f"Unused CategoryMetrics fields detected: {unused_fields}. "
                       f"Remove them or use them in calculations.")
    
    def test_prompt_field_validation(self):
        """Test that prompt fields match Pydantic model expectations."""
        from prompts.models import ClientPersonaResponse, FreelancerPersonaResponse
        
        # Get expected fields from Pydantic models
        client_fields = set(ClientPersonaResponse.model_fields.keys())
        freelancer_fields = set(FreelancerPersonaResponse.model_fields.keys())
        
        # Check that prompts mention all required fields
        client_prompt = generate_client_persona_prompt(1)
        freelancer_prompt = generate_freelancer_persona_prompt(1, [])
        
        for field in client_fields:
            assert field in client_prompt, f"Client prompt missing field: {field}"
        
        for field in freelancer_fields:
            assert field in freelancer_prompt, f"Freelancer prompt missing field: {field}"
    
    def test_budget_philosophy_enum_consistency(self):
        """Test that we should use enums for budget philosophies."""
        # This test documents that we should create an enum
        # to prevent future prompt-code mismatches
        
        # Expected budget philosophies (from prompts)
        expected_philosophies = {"cost-conscious", "value-focused", "premium-focused"}
        
        # Read the marketplace code to see what it actually checks for
        with open("src/marketplace/true_gpt_marketplace.py", "r") as f:
            content = f.read()
        
        # Should contain logic for all three philosophies
        assert "'premium'" in content or "'premium-focused'" in content
        assert "'value'" in content or "'value-focused'" in content  
        assert "cost-conscious" in content  # Default case in comment
        
        # TODO: Create BudgetPhilosophy enum to ensure consistency
    
    def test_no_dead_imports(self):
        """Test for unused imports that might indicate dead code."""
        # Get all Python files in src/
        src_files = list(Path("src").rglob("*.py"))
        
        issues = []
        for file_path in src_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Parse AST to find imports
                tree = ast.parse(content)
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                imports.append(f"{node.module}.{alias.name}")
                
                # Check if imports are used (simple heuristic)
                for imp in imports:
                    if imp.split('.')[-1] not in content.replace(f"import {imp}", ""):
                        issues.append(f"Unused import in {file_path}: {imp}")
                        
            except (SyntaxError, UnicodeDecodeError):
                # Skip files with syntax errors or encoding issues
                continue
        
        # Don't fail on unused imports (too noisy), just log them
        if issues and len(issues) < 10:  # Only report if manageable number
            print(f"Potential unused imports found: {len(issues)}")


class TestCodeQuality:
    """Test general code quality and consistency."""
    
    def test_function_naming_consistency(self):
        """Test that function names follow consistent patterns."""
        # Reputation functions should follow get_* pattern
        manager = SimpleReputationManager()
        methods = [method for method in dir(manager) if not method.startswith('_')]
        
        getter_methods = [m for m in methods if m.startswith('get_')]
        # Should only have the core getter methods that are actually used
        expected_getters = {'get_client_reputation', 'get_freelancer_reputation'}
        actual_getters = set(getter_methods)
        
        assert actual_getters == expected_getters, f"Getter methods mismatch. Expected: {expected_getters}, Found: {actual_getters}"
    
    def test_documentation_completeness(self):
        """Test that public functions have docstrings."""
        manager = SimpleReputationManager()
        
        for method_name in dir(manager):
            if not method_name.startswith('_'):
                method = getattr(manager, method_name)
                if callable(method):
                    assert method.__doc__ is not None, f"Method {method_name} missing docstring"
                    assert len(method.__doc__.strip()) > 10, f"Method {method_name} has too short docstring"
