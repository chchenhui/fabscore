# -*- coding: utf-8 -*-
"""
YuLan-OneSim Experimental Platform

This module provides experimental research capabilities for YuLan-OneSim,
enabling controlled experiments with profile interventions and systematic
result analysis.
"""

from .profile_modifier import ProfileModifier
from .intervention_engine import InterventionEngine
from .config_generator import ConfigGenerator
from .experiment_project import ExperimentProject
from .runtime_intervention import RuntimeInterventionEngine

__all__ = [
    'ProfileModifier',
    'InterventionEngine', 
    'ConfigGenerator',
    'ExperimentProject',
    'RuntimeInterventionEngine',
]

__version__ = '1.0.0'