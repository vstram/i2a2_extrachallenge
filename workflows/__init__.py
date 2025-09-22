"""
Workflows package for orchestrating data analysis pipelines.

This package contains workflow orchestration modules that coordinate
the execution of data processing, analysis, and reporting tasks.
"""

from .analysis_workflow import (
    AnalysisWorkflow,
    WorkflowConfig,
    WorkflowStep,
    WorkflowResult,
    WorkflowError,
    run_fraud_detection_analysis,
    run_quick_analysis
)

__all__ = [
    'AnalysisWorkflow',
    'WorkflowConfig',
    'WorkflowStep',
    'WorkflowResult',
    'WorkflowError',
    'run_fraud_detection_analysis',
    'run_quick_analysis'
]