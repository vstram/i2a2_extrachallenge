"""
Report Agent for Fraud Detection Analysis

This module implements a LangChain-based agent that converts analysis results
from the Analyser Agent into formatted markdown reports with streaming capabilities
and structured outputs suitable for Streamlit display.
"""

import json
import logging
import base64
from typing import Dict, Any, List, Optional, Iterator, Callable
from dataclasses import dataclass
from datetime import datetime

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

# Utils imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_config import LLMManager, LLMProvider, create_llm_manager
from utils.prompts import (
    FraudDetectionPrompts, AnalysisType, AgentRole,
    format_reporter_prompt, get_fraud_prompts
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReportRequest:
    """Request structure for report generation."""
    analysis_results: Dict[str, Any]
    available_charts: Optional[List[str]] = None
    chart_data: Optional[Dict[str, str]] = None  # Base64 encoded charts
    dataset_name: str = "fraud_dataset"
    analysis_timestamp: str = None
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    report_style: str = "comprehensive"  # comprehensive, technical, executive
    stream_callback: Optional[Callable[[str], None]] = None


@dataclass
class ReportResult:
    """Result structure for report generation."""
    report_data: Dict[str, Any]
    markdown_report: str
    processing_time: float
    timestamp: str
    report_style: str
    metadata: Dict[str, Any]


class ReporterAgentError(Exception):
    """Custom exception for Reporter Agent errors."""
    pass


class ReporterAgent:
    """
    LangChain-based Reporter Agent for fraud detection analysis reports.

    Converts analysis results from the Analyser Agent into formatted markdown
    reports with embedded charts, tables, and structured insights optimized
    for Streamlit display.
    """

    def __init__(self, llm_manager: LLMManager = None, prompts: FraudDetectionPrompts = None):
        """
        Initialize the Reporter Agent.

        Args:
            llm_manager: LLM manager instance (will create default if None)
            prompts: Fraud detection prompts instance (will create default if None)
        """
        self.llm_manager = llm_manager or create_llm_manager()
        self.prompts = prompts or get_fraud_prompts()
        self.report_history: List[ReportResult] = []

    def generate_report(self, request: ReportRequest) -> ReportResult:
        """
        Generate a markdown report from analysis results.

        Args:
            request: Report generation request with analysis data and parameters

        Returns:
            ReportResult with structured report data and markdown

        Raises:
            ReporterAgentError: If report generation fails
        """
        start_time = datetime.now()

        try:
            # Validate input data
            self._validate_input_data(request)

            # Prepare prompt data
            prompt_data = self._prepare_prompt_data(request)

            # Get LLM response
            if request.stream_callback:
                report_json = self._generate_with_streaming(prompt_data, request.stream_callback)
            else:
                report_json = self._generate_without_streaming(prompt_data)

            # Parse and validate response
            report_data = self._parse_and_validate_response(report_json)

            # Extract markdown report
            markdown_report = self._extract_markdown_report(report_data, request)

            # Calculate metrics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Create result
            result = ReportResult(
                report_data=report_data,
                markdown_report=markdown_report,
                processing_time=processing_time,
                timestamp=end_time.isoformat(),
                report_style=request.report_style,
                metadata={
                    'dataset_name': request.dataset_name,
                    'analysis_type': request.analysis_type.value,
                    'llm_provider': self.llm_manager.active_provider.value if self.llm_manager.active_provider else 'unknown',
                    'prompt_version': '1.0',
                    'has_charts': bool(request.chart_data),
                    'chart_count': len(request.available_charts) if request.available_charts else 0
                }
            )

            # Store in history
            self.report_history.append(result)

            logger.info(f"Report generated in {processing_time:.2f}s, {len(markdown_report)} characters")
            return result

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise ReporterAgentError(f"Report generation failed: {e}")

    def _validate_input_data(self, request: ReportRequest) -> None:
        """Validate input data structure."""
        if not isinstance(request.analysis_results, dict):
            raise ReporterAgentError("Analysis results must be a dictionary")

        # Check for required fields in analysis results
        required_fields = ['analysis_summary', 'fraud_specific_analysis']
        missing_fields = [field for field in required_fields
                         if field not in request.analysis_results]

        if missing_fields:
            logger.warning(f"Missing recommended fields in analysis results: {missing_fields}")

        # Set default timestamp if not provided
        if request.analysis_timestamp is None:
            request.analysis_timestamp = datetime.now().isoformat()

    def _prepare_prompt_data(self, request: ReportRequest) -> Dict[str, str]:
        """Prepare prompt data for LLM."""
        # Convert analysis results to JSON string
        analysis_json = json.dumps(request.analysis_results, indent=2)

        # Format available charts
        charts_description = self._format_charts_description(request.available_charts, request.chart_data)

        # Format prompt using templates
        return format_reporter_prompt(
            analysis_results=analysis_json,
            available_charts=charts_description,
            dataset_name=request.dataset_name,
            timestamp=request.analysis_timestamp,
            analysis_type_name=request.analysis_type.value
        )

    def _format_charts_description(self, available_charts: Optional[List[str]],
                                 chart_data: Optional[Dict[str, str]]) -> str:
        """Format description of available charts."""
        if not available_charts:
            return "No charts available"

        descriptions = []
        for chart_name in available_charts:
            if chart_data and chart_name in chart_data:
                descriptions.append(f"- {chart_name}: Available as base64 encoded image")
            else:
                descriptions.append(f"- {chart_name}: Chart available")

        return "\n".join(descriptions)

    def _generate_with_streaming(self, prompt_data: Dict[str, str],
                               callback: Callable[[str], None]) -> str:
        """Perform report generation with streaming."""
        messages = [
            SystemMessage(content=prompt_data['system_prompt']),
            HumanMessage(content=prompt_data['user_prompt'])
        ]

        # Stream response
        response_chunks = []
        try:
            for chunk in self.llm_manager.stream_llm(messages):
                if chunk:
                    response_chunks.append(chunk)
                    callback(chunk)  # Real-time feedback

            return ''.join(response_chunks)

        except Exception as e:
            logger.error(f"Streaming report generation failed: {e}")
            # Fallback to non-streaming
            logger.info("Falling back to non-streaming report generation")
            return self._generate_without_streaming(prompt_data)

    def _generate_without_streaming(self, prompt_data: Dict[str, str]) -> str:
        """Perform report generation without streaming."""
        messages = [
            SystemMessage(content=prompt_data['system_prompt']),
            HumanMessage(content=prompt_data['user_prompt'])
        ]

        try:
            # Try with fallback mechanism
            return self.llm_manager.try_providers_with_fallback(messages)
        except Exception as e:
            logger.error(f"All LLM providers failed: {e}")
            raise ReporterAgentError(f"LLM report generation failed: {e}")

    def _parse_and_validate_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = self._extract_json_from_response(response_text)

            # Parse JSON
            report_data = json.loads(json_text)

            # Validate schema
            if not self.prompts.validate_output_schema(report_data, AgentRole.REPORTER):
                logger.warning("Response doesn't match expected schema, proceeding with best effort")

            return report_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Return structured fallback
            return self._create_fallback_report(response_text)

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from response text (handle markdown code blocks)."""
        # Remove markdown code blocks if present
        lines = response_text.split('\n')
        json_lines = []
        in_json_block = False
        json_started = False

        for line in lines:
            # Skip markdown code block markers
            if line.strip().startswith('```'):
                if 'json' in line.lower() or not json_started:
                    in_json_block = not in_json_block
                continue

            # Look for JSON start
            if not json_started and (line.strip().startswith('{') or in_json_block):
                json_started = True

            if json_started:
                json_lines.append(line)

        # Join and clean up
        json_text = '\n'.join(json_lines).strip()

        # If no clear JSON found, try to find JSON-like content
        if not json_text or not json_text.startswith('{'):
            # Look for any JSON-like structure in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0).strip()
            else:
                raise ValueError("No JSON structure found in response")

        # Clean up any trailing text after the JSON
        if json_text.count('{') > 0:
            # Find the matching closing brace for the first opening brace
            brace_count = 0
            end_pos = 0
            for i, char in enumerate(json_text):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break

            if end_pos > 0:
                json_text = json_text[:end_pos]

        return json_text

    def _create_fallback_report(self, response_text: str) -> Dict[str, Any]:
        """Create fallback report when JSON parsing fails."""
        logger.warning("Creating fallback report from text response")

        # Try to extract markdown content if response is mostly markdown
        if response_text.startswith('#') or '##' in response_text:
            markdown_content = response_text
        else:
            markdown_content = f"# Analysis Report\n\n{response_text}"

        return {
            "report_metadata": {
                "title": "Fraud Detection Analysis Report",
                "generated_at": datetime.now().isoformat(),
                "analysis_type": "comprehensive",
                "executive_summary": "Report generated with limited formatting due to response parsing issues"
            },
            "markdown_report": markdown_content,
            "sections": [
                {
                    "section_title": "Raw Analysis Output",
                    "content": response_text[:1000],  # Truncate for safety
                    "charts": [],
                    "tables": []
                }
            ],
            "key_metrics": {
                "fraud_rate": 0.0,
                "total_transactions": 0,
                "data_quality_score": 0.5,
                "analysis_confidence": 0.3
            }
        }

    def _extract_markdown_report(self, report_data: Dict[str, Any],
                                request: ReportRequest) -> str:
        """Extract and enhance markdown report from report data."""
        # Get the main markdown report
        markdown_report = report_data.get('markdown_report', '')

        # If no markdown report, create one from the structured data
        if not markdown_report:
            markdown_report = self._create_markdown_from_structure(report_data, request)

        # Enhance with embedded charts if available
        if request.chart_data:
            markdown_report = self._embed_charts_in_markdown(markdown_report, request.chart_data)

        return markdown_report

    def _create_markdown_from_structure(self, report_data: Dict[str, Any],
                                      request: ReportRequest) -> str:
        """Create markdown report from structured report data."""
        metadata = report_data.get('report_metadata', {})
        sections = report_data.get('sections', [])
        key_metrics = report_data.get('key_metrics', {})

        # Build markdown report
        markdown_parts = []

        # Title and metadata
        title = metadata.get('title', 'Fraud Detection Analysis Report')
        markdown_parts.append(f"# {title}")
        markdown_parts.append("")

        # Executive summary
        if 'executive_summary' in metadata:
            markdown_parts.append("## Executive Summary")
            markdown_parts.append(metadata['executive_summary'])
            markdown_parts.append("")

        # Key metrics
        if key_metrics:
            markdown_parts.append("## Key Metrics")
            for metric_name, metric_value in key_metrics.items():
                if isinstance(metric_value, float):
                    if metric_name in ['fraud_rate', 'data_quality_score', 'analysis_confidence']:
                        formatted_value = f"{metric_value:.2%}" if metric_name == 'fraud_rate' else f"{metric_value:.2f}"
                    else:
                        formatted_value = f"{metric_value:.2f}"
                else:
                    formatted_value = str(metric_value)

                formatted_name = metric_name.replace('_', ' ').title()
                markdown_parts.append(f"- **{formatted_name}**: {formatted_value}")
            markdown_parts.append("")

        # Sections
        for section in sections:
            section_title = section.get('section_title', 'Analysis Section')
            content = section.get('content', '')

            markdown_parts.append(f"## {section_title}")
            markdown_parts.append(content)
            markdown_parts.append("")

        # Add generation info
        generated_at = metadata.get('generated_at', datetime.now().isoformat())
        markdown_parts.append("---")
        markdown_parts.append(f"*Report generated at {generated_at}*")

        return '\n'.join(markdown_parts)

    def _embed_charts_in_markdown(self, markdown_report: str,
                                chart_data: Dict[str, str]) -> str:
        """Embed base64 encoded charts into markdown report."""
        # For each chart, try to find a suitable place to embed it
        for chart_name, chart_base64 in chart_data.items():
            # Create markdown image syntax for base64 data
            chart_markdown = f"\n![{chart_name}](data:image/png;base64,{chart_base64})\n"

            # Try to place chart after relevant section headers
            chart_keywords = {
                'histogram': ['distribution', 'frequency', 'data description'],
                'correlation': ['correlation', 'relationship', 'feature'],
                'scatter': ['scatter', 'relationship', 'pattern'],
                'fraud': ['fraud', 'class', 'target']
            }

            # Find appropriate placement
            chart_type = self._identify_chart_type(chart_name)
            keywords = chart_keywords.get(chart_type, [chart_name.lower()])

            # Look for section headers that match keywords
            lines = markdown_report.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('##') and any(keyword in line.lower() for keyword in keywords):
                    # Insert chart after this header and any immediate content
                    insert_pos = i + 1
                    while (insert_pos < len(lines) and
                           lines[insert_pos].strip() and
                           not lines[insert_pos].startswith('#')):
                        insert_pos += 1

                    lines.insert(insert_pos, chart_markdown)
                    break
            else:
                # If no suitable place found, add at the end before footer
                if '---' in markdown_report:
                    markdown_report = markdown_report.replace('---', f"{chart_markdown}\n---")
                else:
                    markdown_report += chart_markdown
                continue

            markdown_report = '\n'.join(lines)

        return markdown_report

    def _identify_chart_type(self, chart_name: str) -> str:
        """Identify chart type from chart name."""
        chart_name_lower = chart_name.lower()

        if 'fraud' in chart_name_lower or 'class' in chart_name_lower:
            return 'fraud'
        elif 'correlation' in chart_name_lower or 'corr' in chart_name_lower:
            return 'correlation'
        elif 'scatter' in chart_name_lower:
            return 'scatter'
        elif 'histogram' in chart_name_lower or 'dist' in chart_name_lower:
            return 'histogram'
        else:
            return 'other'

    def generate_comprehensive_report(self, analysis_results: Dict[str, Any],
                                    dataset_name: str = "fraud_dataset",
                                    charts: Optional[Dict[str, str]] = None) -> ReportResult:
        """Generate comprehensive fraud detection report."""
        request = ReportRequest(
            analysis_results=analysis_results,
            available_charts=list(charts.keys()) if charts else None,
            chart_data=charts,
            dataset_name=dataset_name,
            analysis_type=AnalysisType.COMPREHENSIVE,
            report_style="comprehensive"
        )
        return self.generate_report(request)

    def generate_technical_report(self, analysis_results: Dict[str, Any],
                                dataset_name: str = "fraud_dataset") -> ReportResult:
        """Generate technical report for data scientists."""
        request = ReportRequest(
            analysis_results=analysis_results,
            dataset_name=dataset_name,
            analysis_type=AnalysisType.COMPREHENSIVE,
            report_style="technical"
        )
        return self.generate_report(request)

    def generate_executive_summary(self, analysis_results: Dict[str, Any],
                                 dataset_name: str = "fraud_dataset") -> ReportResult:
        """Generate executive summary for business stakeholders."""
        request = ReportRequest(
            analysis_results=analysis_results,
            dataset_name=dataset_name,
            analysis_type=AnalysisType.COMPREHENSIVE,
            report_style="executive"
        )
        return self.generate_report(request)

    def generate_streaming_report(self, analysis_results: Dict[str, Any],
                                stream_callback: Callable[[str], None],
                                dataset_name: str = "fraud_dataset",
                                charts: Optional[Dict[str, str]] = None) -> ReportResult:
        """Generate report with streaming output for real-time display."""
        request = ReportRequest(
            analysis_results=analysis_results,
            available_charts=list(charts.keys()) if charts else None,
            chart_data=charts,
            dataset_name=dataset_name,
            analysis_type=AnalysisType.COMPREHENSIVE,
            report_style="comprehensive",
            stream_callback=stream_callback
        )
        return self.generate_report(request)

    def get_report_history(self) -> List[ReportResult]:
        """Get report generation history."""
        return self.report_history.copy()

    def get_latest_report(self) -> Optional[ReportResult]:
        """Get the latest report result."""
        return self.report_history[-1] if self.report_history else None

    def clear_history(self):
        """Clear report history."""
        self.report_history.clear()

    def export_report_to_file(self, report_result: ReportResult,
                            file_path: str, format: str = "markdown") -> None:
        """
        Export report to file.

        Args:
            report_result: Report result to export
            file_path: Output file path
            format: Export format ("markdown", "html", "json")
        """
        try:
            if format.lower() == "markdown":
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_result.markdown_report)
            elif format.lower() == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report_result.report_data, f, indent=2)
            elif format.lower() == "html":
                # Convert markdown to HTML (basic implementation)
                html_content = self._markdown_to_html(report_result.markdown_report)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Report exported to {file_path} in {format} format")

        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            raise ReporterAgentError(f"Export failed: {e}")

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown to basic HTML (simplified implementation)."""
        html_lines = []
        in_code_block = False

        for line in markdown_text.split('\n'):
            line = line.rstrip()

            # Code blocks
            if line.startswith('```'):
                if in_code_block:
                    html_lines.append('</pre>')
                else:
                    html_lines.append('<pre>')
                in_code_block = not in_code_block
                continue

            if in_code_block:
                html_lines.append(line)
                continue

            # Headers
            if line.startswith('# '):
                html_lines.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                html_lines.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                html_lines.append(f'<h3>{line[4:]}</h3>')
            # Bold
            elif '**' in line:
                line = line.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
                html_lines.append(f'<p>{line}</p>')
            # Lists
            elif line.startswith('- '):
                html_lines.append(f'<li>{line[2:]}</li>')
            # Images
            elif line.startswith('!['):
                html_lines.append(line)  # Keep as-is for base64 images
            # Horizontal rule
            elif line.strip() == '---':
                html_lines.append('<hr>')
            # Empty line
            elif not line.strip():
                html_lines.append('<br>')
            # Regular paragraph
            else:
                html_lines.append(f'<p>{line}</p>')

        return '\n'.join(html_lines)

    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and configuration."""
        llm_info = self.llm_manager.get_provider_info()

        return {
            'agent_type': 'reporter',
            'version': '1.0',
            'active_llm_provider': llm_info.get('active_provider'),
            'available_providers': llm_info.get('available_providers', []),
            'report_history_count': len(self.report_history),
            'supported_report_styles': ['comprehensive', 'technical', 'executive'],
            'supported_export_formats': ['markdown', 'html', 'json'],
            'capabilities': [
                'markdown_report_generation',
                'chart_embedding',
                'streaming_responses',
                'structured_json_output',
                'multiple_report_styles',
                'file_export',
                'html_conversion'
            ]
        }


# Convenience functions
def create_reporter_agent(openai_api_key: Optional[str] = None,
                         ollama_base_url: Optional[str] = None) -> ReporterAgent:
    """
    Create a Reporter Agent with specified LLM configuration.

    Args:
        openai_api_key: OpenAI API key
        ollama_base_url: Ollama server URL

    Returns:
        Configured ReporterAgent instance
    """
    llm_manager = create_llm_manager(
        openai_api_key=openai_api_key,
        ollama_base_url=ollama_base_url,
        auto_detect_available=True
    )

    return ReporterAgent(llm_manager=llm_manager)


def generate_fraud_report(analysis_results: Dict[str, Any],
                         dataset_name: str = "fraud_dataset",
                         report_style: str = "comprehensive",
                         charts: Optional[Dict[str, str]] = None,
                         stream_callback: Optional[Callable[[str], None]] = None,
                         openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to generate fraud detection report quickly.

    Args:
        analysis_results: Analysis results from Analyser Agent
        dataset_name: Name of the dataset
        report_style: Style of report (comprehensive, technical, executive)
        charts: Optional base64 encoded charts
        stream_callback: Optional streaming callback
        openai_api_key: Optional OpenAI API key

    Returns:
        Report results dictionary
    """
    agent = create_reporter_agent(openai_api_key=openai_api_key)

    request = ReportRequest(
        analysis_results=analysis_results,
        available_charts=list(charts.keys()) if charts else None,
        chart_data=charts,
        dataset_name=dataset_name,
        analysis_type=AnalysisType.COMPREHENSIVE,
        report_style=report_style,
        stream_callback=stream_callback
    )

    result = agent.generate_report(request)
    return {
        'report_data': result.report_data,
        'markdown_report': result.markdown_report,
        'processing_time': result.processing_time,
        'timestamp': result.timestamp,
        'metadata': result.metadata
    }


if __name__ == "__main__":
    # Example usage and testing
    import sys

    def test_reporter_agent():
        """Test the Reporter Agent with sample data."""
        print("=== Reporter Agent Test ===")

        # Create agent
        try:
            agent = create_reporter_agent()
            print("✓ Agent created successfully")
        except Exception as e:
            print(f"✗ Failed to create agent: {e}")
            return

        # Show agent status
        status = agent.get_agent_status()
        print(f"Agent status: {status['active_llm_provider']} provider active")
        print(f"Capabilities: {len(status['capabilities'])} features")

        # Sample analysis results (from Analyser Agent)
        sample_analysis = {
            "analysis_summary": {
                "dataset_overview": "Credit card fraud dataset with 231 transactions, 0.87% fraud rate",
                "key_findings": [
                    "Highly imbalanced dataset with 2 fraud cases out of 231 transactions",
                    "Transaction amounts range from $0 to $25,691 with high variability",
                    "Clustering analysis shows 3 distinct transaction patterns"
                ],
                "fraud_insights": "Fraudulent transactions show distinct patterns in PCA components",
                "data_quality_assessment": "High quality dataset with no missing values"
            },
            "fraud_specific_analysis": {
                "fraud_rate": 0.0087,
                "temporal_patterns": {"peak_hours": [14, 15, 16]},
                "amount_patterns": {"fraud_avg": 122.65, "normal_avg": 88.03},
                "feature_importance": {"V14": 0.85, "V17": 0.73},
                "anomaly_detection": {"outliers_detected": 380}
            },
            "recommendations": {
                "data_preprocessing": ["Apply SMOTE for class balancing"],
                "feature_engineering": ["Focus on V14 and V17 features"],
                "model_recommendations": ["Use ensemble methods for imbalanced data"],
                "further_analysis": ["Analyze temporal patterns in detail"]
            },
            "confidence_scores": {
                "analysis_confidence": 0.88,
                "data_quality_score": 0.95,
                "pattern_strength": 0.73
            }
        }

        # Test different report styles
        report_styles = ["comprehensive", "technical", "executive"]

        for style in report_styles:
            try:
                print(f"\nTesting {style} report generation...")

                request = ReportRequest(
                    analysis_results=sample_analysis,
                    dataset_name="test_creditcard.csv",
                    analysis_type=AnalysisType.COMPREHENSIVE,
                    report_style=style
                )

                result = agent.generate_report(request)

                print(f"✓ {style} report generated successfully")
                print(f"  Processing time: {result.processing_time:.2f}s")
                print(f"  Report length: {len(result.markdown_report)} characters")
                print(f"  Sections: {len(result.report_data.get('sections', []))}")

                # Show a snippet of the markdown
                snippet = result.markdown_report[:200] + "..." if len(result.markdown_report) > 200 else result.markdown_report
                print(f"  Preview: {snippet}")

            except Exception as e:
                print(f"✗ {style} report generation failed: {e}")

        # Test chart integration
        try:
            print(f"\nTesting report with chart integration...")

            # Mock base64 chart data (just a placeholder)
            mock_charts = {
                "fraud_distribution": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "correlation_heatmap": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            }

            result = agent.generate_comprehensive_report(
                analysis_results=sample_analysis,
                dataset_name="test_with_charts.csv",
                charts=mock_charts
            )

            print("✓ Report with charts generated successfully")
            print(f"  Chart references: {result.markdown_report.count('![')}")

        except Exception as e:
            print(f"✗ Chart integration failed: {e}")

        # Show final status
        final_status = agent.get_agent_status()
        print(f"\nReport history: {final_status['report_history_count']} reports generated")

        print("\n✅ Reporter Agent test completed!")

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_reporter_agent()
    else:
        print("Reporter Agent for Fraud Detection")
        print("Usage: python reporter.py test")
        print("Ensure LLM providers are configured (OpenAI API key or Ollama server)")