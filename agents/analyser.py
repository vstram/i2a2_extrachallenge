"""
Analyser Agent for Fraud Detection Statistics

This module implements a LangChain-based agent that processes statistics JSON
from CSV files and generates comprehensive fraud detection analysis with
streaming capabilities and structured JSON outputs.
"""

import json
import logging
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
    format_analyser_prompt, get_fraud_prompts
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisRequest:
    """Request structure for analysis."""
    statistics_data: Dict[str, Any]
    pattern_data: Dict[str, Any]
    chart_metadata: Optional[Dict[str, Any]] = None
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    dataset_name: str = "fraud_dataset"
    stream_callback: Optional[Callable[[str], None]] = None


@dataclass
class AnalysisResult:
    """Result structure for analysis."""
    analysis_data: Dict[str, Any]
    confidence_score: float
    processing_time: float
    timestamp: str
    analysis_type: AnalysisType
    metadata: Dict[str, Any]


class AnalyserAgentError(Exception):
    """Custom exception for Analyser Agent errors."""
    pass


class AnalyserAgent:
    """
    LangChain-based Analyser Agent for fraud detection statistics analysis.

    Processes statistics JSON from CSV files and generates comprehensive
    analysis with fraud detection insights, streaming capabilities, and
    structured JSON outputs.
    """

    def __init__(self, llm_manager: LLMManager = None, prompts: FraudDetectionPrompts = None):
        """
        Initialize the Analyser Agent.

        Args:
            llm_manager: LLM manager instance (will create default if None)
            prompts: Fraud detection prompts instance (will create default if None)
        """
        self.llm_manager = llm_manager or create_llm_manager()
        self.prompts = prompts or get_fraud_prompts()
        self.analysis_history: List[AnalysisResult] = []

    def analyze_data(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Analyze data according to the request specifications.

        Args:
            request: Analysis request with data and parameters

        Returns:
            AnalysisResult with structured analysis data

        Raises:
            AnalyserAgentError: If analysis fails
        """
        start_time = datetime.now()

        try:
            # Validate input data
            self._validate_input_data(request)

            # Prepare prompt
            prompt_data = self._prepare_prompt_data(request)

            # Get LLM response
            if request.stream_callback:
                analysis_json = self._analyze_with_streaming(prompt_data, request.stream_callback)
            else:
                analysis_json = self._analyze_without_streaming(prompt_data)

            # Parse and validate response
            analysis_data = self._parse_and_validate_response(analysis_json)

            # Calculate metrics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            confidence_score = self._calculate_confidence_score(analysis_data)

            # Create result
            result = AnalysisResult(
                analysis_data=analysis_data,
                confidence_score=confidence_score,
                processing_time=processing_time,
                timestamp=end_time.isoformat(),
                analysis_type=request.analysis_type,
                metadata={
                    'dataset_name': request.dataset_name,
                    'llm_provider': self.llm_manager.active_provider.value if self.llm_manager.active_provider else 'unknown',
                    'prompt_version': '1.0',
                    'has_chart_metadata': request.chart_metadata is not None
                }
            )

            # Store in history
            self.analysis_history.append(result)

            logger.info(f"Analysis completed in {processing_time:.2f}s with confidence {confidence_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise AnalyserAgentError(f"Analysis failed: {e}")

    def _validate_input_data(self, request: AnalysisRequest) -> None:
        """Validate input data structure."""
        if not isinstance(request.statistics_data, dict):
            raise AnalyserAgentError("Statistics data must be a dictionary")

        if not isinstance(request.pattern_data, dict):
            raise AnalyserAgentError("Pattern data must be a dictionary")

        # Check for required fields in statistics
        required_stats_fields = ['file_info']
        for field in required_stats_fields:
            if field not in request.statistics_data:
                logger.warning(f"Missing recommended field in statistics: {field}")

        # Check for required fields in patterns
        if not request.pattern_data:
            logger.warning("Pattern data is empty")

    def _prepare_prompt_data(self, request: AnalysisRequest) -> Dict[str, str]:
        """Prepare prompt data for LLM."""
        # Convert data to JSON strings
        statistics_json = json.dumps(request.statistics_data, indent=2)
        pattern_json = json.dumps(request.pattern_data, indent=2)
        chart_json = json.dumps(request.chart_metadata or {}, indent=2)

        # Format prompt using templates
        return format_analyser_prompt(
            analysis_type=request.analysis_type,
            statistics_data=statistics_json,
            pattern_data=pattern_json,
            chart_metadata=chart_json
        )

    def _analyze_with_streaming(self, prompt_data: Dict[str, str], callback: Callable[[str], None]) -> str:
        """Perform analysis with streaming."""
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
            logger.error(f"Streaming analysis failed: {e}")
            # Fallback to non-streaming
            logger.info("Falling back to non-streaming analysis")
            return self._analyze_without_streaming(prompt_data)

    def _analyze_without_streaming(self, prompt_data: Dict[str, str]) -> str:
        """Perform analysis without streaming."""
        messages = [
            SystemMessage(content=prompt_data['system_prompt']),
            HumanMessage(content=prompt_data['user_prompt'])
        ]

        try:
            # Try with fallback mechanism
            return self.llm_manager.try_providers_with_fallback(messages)
        except Exception as e:
            logger.error(f"All LLM providers failed: {e}")
            raise AnalyserAgentError(f"LLM analysis failed: {e}")

    def _parse_and_validate_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = self._extract_json_from_response(response_text)

            # Parse JSON
            analysis_data = json.loads(json_text)

            # Validate schema
            if not self.prompts.validate_output_schema(analysis_data, AgentRole.ANALYSER):
                logger.warning("Response doesn't match expected schema, proceeding with best effort")

            return analysis_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Return structured fallback
            return self._create_fallback_analysis(response_text)

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
                json_text = json_match.group(0)
            else:
                raise ValueError("No JSON structure found in response")

        return json_text

    def _create_fallback_analysis(self, response_text: str) -> Dict[str, Any]:
        """Create fallback analysis when JSON parsing fails."""
        logger.warning("Creating fallback analysis from text response")

        return {
            "analysis_summary": {
                "dataset_overview": "Analysis completed but response format was unexpected",
                "key_findings": [
                    "Raw analysis response received",
                    "JSON parsing failed - manual review recommended"
                ],
                "fraud_insights": "Review required due to response format issues",
                "data_quality_assessment": "Unable to assess due to parsing issues"
            },
            "fraud_specific_analysis": {
                "fraud_rate": 0.0,
                "temporal_patterns": {},
                "amount_patterns": {},
                "feature_importance": {},
                "anomaly_detection": {}
            },
            "recommendations": {
                "data_preprocessing": ["Review LLM response format"],
                "feature_engineering": ["Manual analysis recommended"],
                "model_recommendations": ["Improve prompt engineering"],
                "further_analysis": ["Check LLM configuration"]
            },
            "confidence_scores": {
                "analysis_confidence": 0.1,
                "data_quality_score": 0.5,
                "pattern_strength": 0.1
            },
            "raw_response": response_text[:1000]  # Include sample of raw response
        }

    def _calculate_confidence_score(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis."""
        try:
            # Extract confidence scores if available
            confidence_scores = analysis_data.get('confidence_scores', {})

            if confidence_scores:
                # Average the available confidence scores
                scores = [
                    confidence_scores.get('analysis_confidence', 0.5),
                    confidence_scores.get('data_quality_score', 0.5),
                    confidence_scores.get('pattern_strength', 0.5)
                ]
                return sum(scores) / len(scores)
            else:
                # Calculate heuristic confidence based on completeness
                completeness_score = self._calculate_completeness_score(analysis_data)
                return min(max(completeness_score, 0.0), 1.0)

        except Exception as e:
            logger.warning(f"Error calculating confidence score: {e}")
            return 0.5  # Default moderate confidence

    def _calculate_completeness_score(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate completeness score based on presence of key sections."""
        required_sections = [
            'analysis_summary',
            'fraud_specific_analysis',
            'recommendations'
        ]

        present_sections = sum(1 for section in required_sections if section in analysis_data)
        base_score = present_sections / len(required_sections)

        # Bonus for detailed sections
        bonus = 0.0
        if 'analysis_summary' in analysis_data:
            summary = analysis_data['analysis_summary']
            if isinstance(summary.get('key_findings'), list) and len(summary['key_findings']) > 0:
                bonus += 0.1

        if 'recommendations' in analysis_data:
            recommendations = analysis_data['recommendations']
            if isinstance(recommendations, dict) and len(recommendations) > 0:
                bonus += 0.1

        return min(base_score + bonus, 1.0)

    def analyze_data_description(self, statistics_data: Dict[str, Any], pattern_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze data description aspects."""
        request = AnalysisRequest(
            statistics_data=statistics_data,
            pattern_data=pattern_data,
            analysis_type=AnalysisType.DATA_DESCRIPTION
        )
        return self.analyze_data(request)

    def analyze_patterns(self, statistics_data: Dict[str, Any], pattern_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze patterns and trends."""
        request = AnalysisRequest(
            statistics_data=statistics_data,
            pattern_data=pattern_data,
            analysis_type=AnalysisType.PATTERN_IDENTIFICATION
        )
        return self.analyze_data(request)

    def analyze_outliers(self, statistics_data: Dict[str, Any], pattern_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze outliers and their impact."""
        request = AnalysisRequest(
            statistics_data=statistics_data,
            pattern_data=pattern_data,
            analysis_type=AnalysisType.OUTLIER_DETECTION
        )
        return self.analyze_data(request)

    def analyze_relationships(self, statistics_data: Dict[str, Any], pattern_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze variable relationships."""
        request = AnalysisRequest(
            statistics_data=statistics_data,
            pattern_data=pattern_data,
            analysis_type=AnalysisType.RELATIONSHIP_ANALYSIS
        )
        return self.analyze_data(request)

    def analyze_fraud_patterns(self, statistics_data: Dict[str, Any], pattern_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze fraud-specific patterns."""
        request = AnalysisRequest(
            statistics_data=statistics_data,
            pattern_data=pattern_data,
            analysis_type=AnalysisType.FRAUD_ANALYSIS
        )
        return self.analyze_data(request)

    def get_analysis_history(self) -> List[AnalysisResult]:
        """Get analysis history."""
        return self.analysis_history.copy()

    def get_latest_analysis(self) -> Optional[AnalysisResult]:
        """Get the latest analysis result."""
        return self.analysis_history[-1] if self.analysis_history else None

    def clear_history(self):
        """Clear analysis history."""
        self.analysis_history.clear()

    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and configuration."""
        llm_info = self.llm_manager.get_provider_info()

        return {
            'agent_type': 'analyser',
            'version': '1.0',
            'active_llm_provider': llm_info.get('active_provider'),
            'available_providers': llm_info.get('available_providers', []),
            'analysis_history_count': len(self.analysis_history),
            'supported_analysis_types': [t.value for t in AnalysisType],
            'capabilities': [
                'data_type_analysis',
                'distribution_insights',
                'pattern_identification',
                'outlier_detection',
                'relationship_analysis',
                'fraud_pattern_analysis',
                'streaming_responses',
                'structured_json_output'
            ]
        }


# Convenience functions
def create_analyser_agent(openai_api_key: Optional[str] = None,
                         ollama_base_url: Optional[str] = None) -> AnalyserAgent:
    """
    Create an Analyser Agent with specified LLM configuration.

    Args:
        openai_api_key: OpenAI API key
        ollama_base_url: Ollama server URL

    Returns:
        Configured AnalyserAgent instance
    """
    llm_manager = create_llm_manager(
        openai_api_key=openai_api_key,
        ollama_base_url=ollama_base_url,
        auto_detect_available=True
    )

    return AnalyserAgent(llm_manager=llm_manager)


def analyze_csv_data(statistics_data: Dict[str, Any],
                    pattern_data: Dict[str, Any],
                    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE,
                    dataset_name: str = "fraud_dataset",
                    stream_callback: Optional[Callable[[str], None]] = None,
                    openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze CSV data quickly.

    Args:
        statistics_data: Statistics JSON from CSV processor
        pattern_data: Pattern analysis JSON from pattern analyzer
        analysis_type: Type of analysis to perform
        dataset_name: Name of the dataset
        stream_callback: Optional streaming callback
        openai_api_key: Optional OpenAI API key

    Returns:
        Analysis results dictionary
    """
    agent = create_analyser_agent(openai_api_key=openai_api_key)

    request = AnalysisRequest(
        statistics_data=statistics_data,
        pattern_data=pattern_data,
        analysis_type=analysis_type,
        dataset_name=dataset_name,
        stream_callback=stream_callback
    )

    result = agent.analyze_data(request)
    return {
        'analysis': result.analysis_data,
        'confidence': result.confidence_score,
        'processing_time': result.processing_time,
        'timestamp': result.timestamp,
        'metadata': result.metadata
    }


if __name__ == "__main__":
    # Example usage and testing
    import sys
    import json

    def test_analyser_agent():
        """Test the Analyser Agent with sample data."""
        print("=== Analyser Agent Test ===")

        # Create agent
        try:
            agent = create_analyser_agent()
            print("✓ Agent created successfully")
        except Exception as e:
            print(f"✗ Failed to create agent: {e}")
            return

        # Show agent status
        status = agent.get_agent_status()
        print(f"Agent status: {status['active_llm_provider']} provider active")
        print(f"Capabilities: {len(status['capabilities'])} features")

        # Sample data (minimal for testing)
        sample_statistics = {
            "file_info": {"total_rows": 231, "total_columns": 31},
            "fraud_specific_analysis": {
                "class_analysis": {
                    "fraud_rate": 0.0087,
                    "fraud_count": 2,
                    "normal_count": 229
                }
            },
            "basic_statistics": {
                "numeric_statistics": {
                    "Amount": {"mean": 88.35, "std": 250.12, "min": 0.0, "max": 25691.16}
                }
            }
        }

        sample_patterns = {
            "clustering_analysis": {
                "optimal_clustering": {"optimal_k": 3, "silhouette_score": 0.45}
            },
            "outlier_analysis": {
                "summary": {"total_outliers_iqr_method": 380}
            },
            "pattern_summary": {
                "key_findings": ["Clustering separates fraud effectively", "380 outliers detected"]
            }
        }

        # Test different analysis types
        analysis_types = [
            AnalysisType.COMPREHENSIVE,
            AnalysisType.DATA_DESCRIPTION,
            AnalysisType.FRAUD_ANALYSIS
        ]

        for analysis_type in analysis_types:
            try:
                print(f"\nTesting {analysis_type.value} analysis...")

                request = AnalysisRequest(
                    statistics_data=sample_statistics,
                    pattern_data=sample_patterns,
                    analysis_type=analysis_type,
                    dataset_name="test_creditcard.csv"
                )

                result = agent.analyze_data(request)

                print(f"✓ {analysis_type.value} analysis completed")
                print(f"  Confidence: {result.confidence_score:.2f}")
                print(f"  Processing time: {result.processing_time:.2f}s")
                print(f"  Key findings: {len(result.analysis_data.get('analysis_summary', {}).get('key_findings', []))}")

            except Exception as e:
                print(f"✗ {analysis_type.value} analysis failed: {e}")

        # Show final status
        final_status = agent.get_agent_status()
        print(f"\nAnalysis history: {final_status['analysis_history_count']} analyses completed")

        print("\n✅ Analyser Agent test completed!")

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_analyser_agent()
    else:
        print("Analyser Agent for Fraud Detection")
        print("Usage: python analyser.py test")
        print("Ensure LLM providers are configured (OpenAI API key or Ollama server)")