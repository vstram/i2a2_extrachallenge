"""
Utils package for Streamlit AI Analytics Chatbot.

This package contains utility modules for processing CSV files,
generating statistics, creating visualizations, performing pattern analysis,
configuring LLM providers, managing prompts, and supporting the AI agent workflow.
"""

from .csv_processor import CSVProcessor, process_csv_file
from .chart_generator import ChartGenerator, generate_charts_for_csv
from .pattern_analyzer import PatternAnalyzer, analyze_patterns_for_csv
from .llm_config import LLMManager, LLMProvider, create_llm_manager, quick_llm_response
from .prompts import (
    FraudDetectionPrompts, AnalysisType, AgentRole,
    format_analyser_prompt, format_reporter_prompt, get_fraud_prompts
)

__all__ = [
    'CSVProcessor', 'process_csv_file',
    'ChartGenerator', 'generate_charts_for_csv',
    'PatternAnalyzer', 'analyze_patterns_for_csv',
    'LLMManager', 'LLMProvider', 'create_llm_manager', 'quick_llm_response',
    'FraudDetectionPrompts', 'AnalysisType', 'AgentRole',
    'format_analyser_prompt', 'format_reporter_prompt', 'get_fraud_prompts'
]