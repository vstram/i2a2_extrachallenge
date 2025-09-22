"""
Analysis Workflow for Fraud Detection Data Processing

This module orchestrates the complete analysis flow from CSV processing through
the Analyser Agent to the Report Agent, with comprehensive error handling,
progress tracking, and user feedback.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import traceback

# Import required modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.csv_processor import CSVProcessor, process_csv_file
from utils.pattern_analyzer import PatternAnalyzer, analyze_patterns_for_csv
from utils.chart_generator import ChartGenerator
from agents.analyser import AnalyserAgent, AnalysisRequest, AnalysisResult, create_analyser_agent
from agents.reporter import ReporterAgent, ReportRequest, ReportResult, create_reporter_agent
from utils.llm_config import LLMManager, create_llm_manager
from utils.prompts import AnalysisType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for the analysis workflow."""
    chunk_size: int = 10000
    max_samples_pattern_analysis: int = 50000
    enable_charts: bool = True
    enable_caching: bool = True
    cache_dir: str = "cache"
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    report_style: str = "comprehensive"
    timeout_seconds: int = 300
    retry_attempts: int = 3
    openai_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None


@dataclass
class WorkflowStep:
    """Represents a single workflow step."""
    name: str
    status: str = "pending"  # pending, in_progress, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    result_size: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkflowResult:
    """Complete workflow execution result."""
    success: bool
    workflow_id: str
    total_duration_seconds: float
    steps: List[WorkflowStep]
    csv_statistics: Optional[Dict[str, Any]] = None
    pattern_analysis: Optional[Dict[str, Any]] = None
    chart_data: Optional[Dict[str, str]] = None
    analysis_result: Optional[AnalysisResult] = None
    report_result: Optional[ReportResult] = None
    final_report_markdown: Optional[str] = None
    error_summary: Optional[str] = None
    cache_hits: int = 0


class WorkflowError(Exception):
    """Custom exception for workflow errors."""
    pass


class AnalysisWorkflow:
    """
    Orchestrates the complete analysis workflow:
    CSV Processing → Pattern Analysis → Chart Generation → Analyser Agent → Report Agent

    Features:
    - Progress tracking and user feedback
    - Error handling and recovery
    - Caching for processed results
    - Streaming callbacks for real-time updates
    - Retry mechanisms
    - Logging for debugging
    """

    def __init__(self, config: WorkflowConfig = None):
        """
        Initialize the analysis workflow.

        Args:
            config: Workflow configuration (uses defaults if None)
        """
        self.config = config or WorkflowConfig()
        self.workflow_id = None
        self.progress_callback: Optional[Callable[[str, float], None]] = None
        self.stream_callback: Optional[Callable[[str], None]] = None

        # Initialize components
        self.csv_processor = CSVProcessor(chunk_size=self.config.chunk_size)
        self.pattern_analyzer = PatternAnalyzer(max_samples=self.config.max_samples_pattern_analysis)
        self.chart_generator = ChartGenerator() if self.config.enable_charts else None

        # LLM manager and agents (initialized lazily)
        self._llm_manager: Optional[LLMManager] = None
        self._analyser_agent: Optional[AnalyserAgent] = None
        self._reporter_agent: Optional[ReporterAgent] = None

        # Cache setup
        if self.config.enable_caching:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)

        # Workflow state
        self.current_steps: List[WorkflowStep] = []
        self.start_time: Optional[datetime] = None

    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set callback for progress updates. Args: (step_name, progress_percentage)"""
        self.progress_callback = callback

    def set_stream_callback(self, callback: Callable[[str], None]):
        """Set callback for streaming text output. Args: (text_chunk)"""
        self.stream_callback = callback

    @property
    def llm_manager(self) -> LLMManager:
        """Get or create LLM manager."""
        if self._llm_manager is None:
            self._llm_manager = create_llm_manager(
                openai_api_key=self.config.openai_api_key,
                ollama_base_url=self.config.ollama_base_url,
                auto_detect_available=True
            )
        return self._llm_manager

    @property
    def analyser_agent(self) -> AnalyserAgent:
        """Get or create Analyser Agent."""
        if self._analyser_agent is None:
            self._analyser_agent = AnalyserAgent(llm_manager=self.llm_manager)
        return self._analyser_agent

    @property
    def reporter_agent(self) -> ReporterAgent:
        """Get or create Reporter Agent."""
        if self._reporter_agent is None:
            self._reporter_agent = ReporterAgent(llm_manager=self.llm_manager)
        return self._reporter_agent

    def run_complete_analysis(self, csv_file_path: str, dataset_name: str = None) -> WorkflowResult:
        """
        Run the complete analysis workflow.

        Args:
            csv_file_path: Path to the CSV file to analyze
            dataset_name: Optional name for the dataset (inferred from filename if None)

        Returns:
            WorkflowResult with all analysis results and metadata

        Raises:
            WorkflowError: If workflow fails critically
        """
        self.workflow_id = f"workflow_{int(time.time())}"
        self.start_time = datetime.now()
        self.current_steps = []

        if dataset_name is None:
            dataset_name = Path(csv_file_path).stem

        logger.info(f"Starting analysis workflow {self.workflow_id} for {dataset_name}")

        try:
            # Step 1: CSV Processing
            step_1 = self._execute_step(
                "csv_processing",
                self._step_process_csv,
                csv_file_path
            )

            # Step 2: Pattern Analysis
            step_2 = self._execute_step(
                "pattern_analysis",
                self._step_analyze_patterns,
                csv_file_path
            )

            # Step 3: Chart Generation (optional)
            step_3 = None
            if self.config.enable_charts:
                step_3 = self._execute_step(
                    "chart_generation",
                    self._step_generate_charts,
                    step_1, step_2
                )

            # Step 4: Analyser Agent
            step_4 = self._execute_step(
                "analyser_agent",
                self._step_run_analyser,
                step_1, step_2, step_3, dataset_name
            )

            # Step 5: Reporter Agent
            step_5 = self._execute_step(
                "reporter_agent",
                self._step_run_reporter,
                step_4, step_3, dataset_name
            )

            # Create final result
            total_duration = (datetime.now() - self.start_time).total_seconds()

            result = WorkflowResult(
                success=True,
                workflow_id=self.workflow_id,
                total_duration_seconds=total_duration,
                steps=self.current_steps,
                csv_statistics=getattr(step_1, 'result', None),
                pattern_analysis=getattr(step_2, 'result', None),
                chart_data=getattr(step_3, 'result', None) if step_3 else None,
                analysis_result=step_4.result if hasattr(step_4, 'result') else None,
                report_result=step_5.result if hasattr(step_5, 'result') else None,
                final_report_markdown=step_5.result.markdown_report if hasattr(step_5, 'result') and step_5.result else None
            )

            logger.info(f"Workflow {self.workflow_id} completed successfully in {total_duration:.2f}s")
            self._notify_progress("Workflow completed", 100.0)

            return result

        except Exception as e:
            error_msg = f"Workflow {self.workflow_id} failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            # Create error result
            total_duration = (datetime.now() - self.start_time).total_seconds()

            result = WorkflowResult(
                success=False,
                workflow_id=self.workflow_id,
                total_duration_seconds=total_duration,
                steps=self.current_steps,
                error_summary=error_msg
            )

            raise WorkflowError(error_msg) from e

    def _execute_step(self, step_name: str, step_function: Callable, *args) -> WorkflowStep:
        """Execute a workflow step with error handling and progress tracking."""
        step = WorkflowStep(name=step_name)
        self.current_steps.append(step)

        logger.info(f"Starting step: {step_name}")
        self._notify_progress(f"Starting {step_name}", len(self.current_steps) * 20 - 20)

        step.status = "in_progress"
        step.start_time = datetime.now()

        try:
            # Execute step with retries
            result = self._execute_with_retry(step_function, *args)

            step.end_time = datetime.now()
            step.duration_seconds = (step.end_time - step.start_time).total_seconds()
            step.status = "completed"
            step.result = result

            # Calculate result size if applicable
            if hasattr(result, '__len__'):
                step.result_size = len(result)
            elif isinstance(result, (dict, list)):
                step.result_size = len(str(result))

            logger.info(f"Step {step_name} completed in {step.duration_seconds:.2f}s")
            self._notify_progress(f"Completed {step_name}", len(self.current_steps) * 20)

            return step

        except Exception as e:
            step.end_time = datetime.now()
            step.duration_seconds = (step.end_time - step.start_time).total_seconds()
            step.status = "failed"
            step.error_message = str(e)

            logger.error(f"Step {step_name} failed after {step.duration_seconds:.2f}s: {e}")
            raise WorkflowError(f"Step {step_name} failed: {e}") from e

    def _execute_with_retry(self, func: Callable, *args) -> Any:
        """Execute function with retry mechanism."""
        last_exception = None

        for attempt in range(self.config.retry_attempts):
            try:
                return func(*args)
            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.config.retry_attempts} attempts failed")

        raise last_exception

    def _step_process_csv(self, csv_file_path: str) -> Dict[str, Any]:
        """Step 1: Process CSV file and generate statistics."""
        cache_key = f"csv_stats_{Path(csv_file_path).name}"

        # Check cache
        if self.config.enable_caching:
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                logger.info("Using cached CSV statistics")
                return cached_result

        # Process CSV
        logger.info(f"Processing CSV file: {csv_file_path}")
        statistics = self.csv_processor.load_and_process(csv_file_path)

        # Convert numpy types to ensure JSON serialization compatibility
        statistics = self._convert_numpy_types(statistics)

        # Cache result
        if self.config.enable_caching:
            self._save_to_cache(cache_key, statistics)

        return statistics

    def _step_analyze_patterns(self, csv_file_path: str) -> Dict[str, Any]:
        """Step 2: Analyze patterns in the data."""
        cache_key = f"patterns_{Path(csv_file_path).name}"

        # Check cache
        if self.config.enable_caching:
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                logger.info("Using cached pattern analysis")
                return cached_result

        # Analyze patterns
        logger.info(f"Analyzing patterns in: {csv_file_path}")
        patterns = self.pattern_analyzer.analyze_all_patterns(
            self._load_csv_sample(csv_file_path)
        )

        # Convert numpy types to ensure JSON serialization compatibility
        patterns = self._convert_numpy_types(patterns)

        # Cache result
        if self.config.enable_caching:
            self._save_to_cache(cache_key, patterns)

        return patterns

    def _step_generate_charts(self, csv_stats: WorkflowStep, pattern_analysis: WorkflowStep) -> Dict[str, str]:
        """Step 3: Generate charts from statistics and patterns."""
        if not self.chart_generator:
            return {}

        logger.info("Generating charts from analysis results")

        try:
            # Generate charts based on statistics and patterns
            charts = self.chart_generator.generate_all_charts(
                csv_stats.result,
                pattern_analysis.result
            )

            logger.info(f"Generated {len(charts)} charts")
            return charts

        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")
            return {}

    def _step_run_analyser(self, csv_stats: WorkflowStep, patterns: WorkflowStep,
                          charts: Optional[WorkflowStep], dataset_name: str) -> AnalysisResult:
        """Step 4: Run Analyser Agent."""
        logger.info("Running Analyser Agent")

        chart_metadata = None
        if charts and charts.result:
            chart_metadata = {name: "chart_available" for name in charts.result.keys()}

        request = AnalysisRequest(
            statistics_data=csv_stats.result,
            pattern_data=patterns.result,
            chart_metadata=chart_metadata,
            analysis_type=self.config.analysis_type,
            dataset_name=dataset_name,
            stream_callback=self.stream_callback
        )

        return self.analyser_agent.analyze_data(request)

    def _step_run_reporter(self, analysis_result: WorkflowStep, charts: Optional[WorkflowStep],
                          dataset_name: str) -> ReportResult:
        """Step 5: Run Reporter Agent."""
        logger.info("Running Reporter Agent")

        chart_data = None
        available_charts = None
        if charts and charts.result:
            chart_data = charts.result
            available_charts = list(charts.result.keys())

        request = ReportRequest(
            analysis_results=analysis_result.result.analysis_data,
            available_charts=available_charts,
            chart_data=chart_data,
            dataset_name=dataset_name,
            analysis_timestamp=analysis_result.result.timestamp,
            analysis_type=self.config.analysis_type,
            report_style=self.config.report_style,
            stream_callback=self.stream_callback
        )

        return self.reporter_agent.generate_report(request)

    def _load_csv_sample(self, csv_file_path: str):
        """Load CSV sample for pattern analysis."""
        import pandas as pd

        # Load a sample for pattern analysis
        if self.config.max_samples_pattern_analysis:
            return pd.read_csv(csv_file_path, nrows=self.config.max_samples_pattern_analysis)
        else:
            return pd.read_csv(csv_file_path)

    def _notify_progress(self, step_name: str, progress: float):
        """Notify progress callback if available."""
        if self.progress_callback:
            try:
                self.progress_callback(step_name, progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load result from cache."""
        if not self.config.enable_caching:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")

        return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]):
        """Save result to cache."""
        if not self.config.enable_caching:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        total_steps = len(self.current_steps)
        completed_steps = sum(1 for step in self.current_steps if step.status == "completed")
        failed_steps = sum(1 for step in self.current_steps if step.status == "failed")

        return {
            'workflow_id': self.workflow_id,
            'total_steps': total_steps,
            'completed_steps': completed_steps,
            'failed_steps': failed_steps,
            'current_step': self.current_steps[-1].name if self.current_steps else None,
            'progress_percentage': (completed_steps / max(total_steps, 1)) * 100,
            'is_running': any(step.status == "in_progress" for step in self.current_steps),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'steps': [
                {
                    'name': step.name,
                    'status': step.status,
                    'duration_seconds': step.duration_seconds,
                    'error_message': step.error_message
                }
                for step in self.current_steps
            ]
        }

    def clear_cache(self):
        """Clear all cached results."""
        if self.config.enable_caching and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    logger.info(f"Removed cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")


# Convenience functions
def run_fraud_detection_analysis(csv_file_path: str,
                                dataset_name: str = None,
                                openai_api_key: str = None,
                                ollama_base_url: str = None,
                                enable_charts: bool = True,
                                progress_callback: Callable[[str, float], None] = None,
                                stream_callback: Callable[[str], None] = None) -> WorkflowResult:
    """
    Convenience function to run complete fraud detection analysis.

    Args:
        csv_file_path: Path to CSV file
        dataset_name: Name of dataset (inferred if None)
        openai_api_key: OpenAI API key
        ollama_base_url: Ollama server URL
        enable_charts: Whether to generate charts
        progress_callback: Optional progress callback
        stream_callback: Optional streaming callback

    Returns:
        WorkflowResult with complete analysis results
    """
    config = WorkflowConfig(
        openai_api_key=openai_api_key,
        ollama_base_url=ollama_base_url,
        enable_charts=enable_charts
    )

    workflow = AnalysisWorkflow(config)

    if progress_callback:
        workflow.set_progress_callback(progress_callback)

    if stream_callback:
        workflow.set_stream_callback(stream_callback)

    return workflow.run_complete_analysis(csv_file_path, dataset_name)


def run_quick_analysis(csv_file_path: str,
                      openai_api_key: str = None) -> str:
    """
    Quick analysis that returns just the markdown report.

    Args:
        csv_file_path: Path to CSV file
        openai_api_key: OpenAI API key

    Returns:
        Markdown report string
    """
    result = run_fraud_detection_analysis(
        csv_file_path=csv_file_path,
        openai_api_key=openai_api_key,
        enable_charts=False
    )

    if result.success and result.final_report_markdown:
        return result.final_report_markdown
    else:
        return f"Analysis failed: {result.error_summary}"


if __name__ == "__main__":
    # Example usage and testing
    import sys

    def test_workflow():
        """Test the analysis workflow with sample data."""
        print("=== Analysis Workflow Test ===")

        # Test with small dataset
        test_csv_path = "data/creditcard-tiny.csv"

        if not Path(test_csv_path).exists():
            print(f"Test file not found: {test_csv_path}")
            print("Please ensure test data is available")
            return

        # Progress callback
        def progress_callback(step_name: str, progress: float):
            print(f"Progress: {step_name} - {progress:.1f}%")

        # Stream callback
        def stream_callback(text: str):
            print(f"Stream: {text[:50]}..." if len(text) > 50 else f"Stream: {text}")

        try:
            config = WorkflowConfig(
                enable_charts=False,  # Disable charts for testing
                enable_caching=True,
                analysis_type=AnalysisType.COMPREHENSIVE
            )

            workflow = AnalysisWorkflow(config)
            workflow.set_progress_callback(progress_callback)
            workflow.set_stream_callback(stream_callback)

            print(f"Starting analysis of {test_csv_path}")
            result = workflow.run_complete_analysis(test_csv_path, "test_dataset")

            print(f"\n✅ Analysis completed successfully!")
            print(f"Workflow ID: {result.workflow_id}")
            print(f"Total duration: {result.total_duration_seconds:.2f}s")
            print(f"Steps completed: {len([s for s in result.steps if s.status == 'completed'])}/{len(result.steps)}")

            if result.final_report_markdown:
                print(f"Report length: {len(result.final_report_markdown)} characters")
                print("Report preview:")
                print(result.final_report_markdown[:500] + "..." if len(result.final_report_markdown) > 500 else result.final_report_markdown)

        except Exception as e:
            print(f"✗ Workflow test failed: {e}")

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_workflow()
    elif len(sys.argv) > 1:
        # Run analysis on provided file
        csv_path = sys.argv[1]
        try:
            print(f"Running analysis on {csv_path}")
            result = run_fraud_detection_analysis(csv_path)
            if result.success:
                print("✅ Analysis completed successfully!")
                if result.final_report_markdown:
                    print("\n" + "="*50)
                    print("ANALYSIS REPORT")
                    print("="*50)
                    print(result.final_report_markdown)
            else:
                print(f"✗ Analysis failed: {result.error_summary}")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("Analysis Workflow for Fraud Detection")
        print("Usage:")
        print("  python analysis_workflow.py test           # Run test")
        print("  python analysis_workflow.py <csv_file>     # Analyze CSV file")