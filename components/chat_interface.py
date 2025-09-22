"""
Chat Interface for AI Analytics Chatbot

This module implements a comprehensive chat interface with message history,
streaming message display, markdown rendering, chart integration, and
agent response handling for the AI analytics chatbot.
"""

import streamlit as st
import base64
import time
import re
import os
import sys
import logging
from typing import Dict, Any, Optional, List, Callable, Generator
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from agents.analyser import AnalyserAgent, AnalysisRequest, AnalysisResult
from agents.reporter import ReporterAgent, ReportRequest, ReportResult
from utils.prompts import AnalysisType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of chat messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ERROR = "error"


class MessageRole(Enum):
    """Message roles for different response types."""
    ANALYSIS = "analysis"
    REPORT = "report"
    CONVERSATION = "conversation"
    HELP = "help"


@dataclass
class ChatMessage:
    """Structure for chat messages."""
    content: str
    role: MessageType
    timestamp: str
    message_id: str
    metadata: Dict[str, Any] = None
    charts: List[str] = None
    streaming: bool = False


class ChatInterface:
    """
    Comprehensive chat interface for AI analytics chatbot.

    Handles message history, streaming responses, markdown rendering,
    chart display, and integration with Analyser and Reporter agents.
    """

    def __init__(self,
                 analyser_agent: Optional[AnalyserAgent] = None,
                 reporter_agent: Optional[ReporterAgent] = None,
                 enable_streaming: bool = True):
        """
        Initialize the chat interface.

        Args:
            analyser_agent: Analyser agent instance
            reporter_agent: Reporter agent instance
            enable_streaming: Whether to enable streaming responses
        """
        self.analyser_agent = analyser_agent
        self.reporter_agent = reporter_agent
        self.enable_streaming = enable_streaming
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables for chat interface."""
        defaults = {
            # Chat state
            'chat_messages': [],
            'chat_history': [],
            'current_conversation_id': None,
            'message_counter': 0,

            # Processing state
            'processing_query': False,
            'streaming_active': False,
            'current_response': '',

            # Agent integration
            'last_analysis_result': None,
            'last_report_result': None,
            'available_data': False,

            # UI state
            'show_message_metadata': False,
            'auto_scroll': True,
            'chat_input_key': 0
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def render(self) -> None:
        """Render the complete chat interface."""
        st.subheader("💬 Chat Interface")

        # Check if data is available
        self._check_data_availability()

        # Render chat messages
        self._render_message_history()

        # Render input area
        self._render_chat_input()

        # Show chat controls
        self._render_chat_controls()

    def _check_data_availability(self) -> None:
        """Check if processed data is available for analysis."""
        has_statistics = st.session_state.get('statistics_data') is not None
        has_patterns = st.session_state.get('pattern_data') is not None
        has_charts = st.session_state.get('chart_data') is not None

        st.session_state.available_data = has_statistics and has_patterns

        if not st.session_state.available_data:
            if not st.session_state.chat_messages:
                self._add_system_message(
                    "👋 Welcome! Please upload a CSV file to begin analyzing your data.",
                    show_suggestions=True
                )

    def _render_message_history(self) -> None:
        """Render the chat message history."""
        # Create container for messages
        message_container = st.container()

        with message_container:
            if not st.session_state.chat_messages:
                # Show welcome message if no messages
                st.info("💡 **Tips to get started:**\n"
                       "- Upload a CSV file using the file uploader above\n"
                       "- Ask questions like 'Analyze my data' or 'Generate a report'\n"
                       "- Request specific analysis types: patterns, outliers, correlations")
            else:
                # Display each message
                for i, message in enumerate(st.session_state.chat_messages):
                    self._render_single_message(message, i)

    def _render_single_message(self, message: ChatMessage, index: int) -> None:
        """
        Render a single chat message.

        Args:
            message: ChatMessage to render
            index: Message index in the list
        """
        # Choose appropriate chat message role for Streamlit
        if message.role == MessageType.USER:
            with st.chat_message("user"):
                st.markdown(message.content)

                # Show metadata if enabled
                if st.session_state.show_message_metadata:
                    with st.expander("Message Details", expanded=False):
                        st.json({
                            'timestamp': message.timestamp,
                            'message_id': message.message_id,
                            'metadata': message.metadata or {}
                        })

        elif message.role == MessageType.ASSISTANT:
            with st.chat_message("assistant"):
                # Render markdown content
                self._render_markdown_content(message.content)

                # Render charts if available
                if message.charts:
                    self._render_message_charts(message.charts)

                # Show metadata if enabled
                if st.session_state.show_message_metadata:
                    with st.expander("Response Details", expanded=False):
                        st.json({
                            'timestamp': message.timestamp,
                            'message_id': message.message_id,
                            'metadata': message.metadata or {},
                            'charts_count': len(message.charts) if message.charts else 0
                        })

        elif message.role == MessageType.SYSTEM:
            with st.chat_message("assistant", avatar="🤖"):
                st.info(message.content)

        elif message.role == MessageType.ERROR:
            with st.chat_message("assistant", avatar="⚠️"):
                st.error(message.content)

    def _render_markdown_content(self, content: str) -> None:
        """
        Render markdown content with special handling for embedded images.

        Args:
            content: Markdown content to render
        """
        # Check for base64 images in markdown
        if "data:image" in content:
            # Split content by image tags
            parts = re.split(r'(!\[.*?\]\(data:image/[^)]+\))', content)

            for part in parts:
                if part.startswith('![') and 'data:image' in part:
                    # Extract image data
                    img_match = re.match(r'!\[(.*?)\]\(data:image/([^;]+);base64,([^)]+)\)', part)
                    if img_match:
                        alt_text, img_format, img_data = img_match.groups()
                        try:
                            # Display image
                            st.image(
                                base64.b64decode(img_data),
                                caption=alt_text,
                                use_column_width=True
                            )
                        except Exception as e:
                            st.error(f"Error displaying image {alt_text}: {e}")
                else:
                    # Regular markdown
                    if part.strip():
                        st.markdown(part)
        else:
            # Regular markdown content
            st.markdown(content)

    def _render_message_charts(self, charts: List[str]) -> None:
        """
        Render charts associated with a message.

        Args:
            charts: List of chart identifiers or base64 data
        """
        if not charts:
            return

        st.markdown("### 📊 Generated Charts")

        # Create columns for charts
        if len(charts) == 1:
            cols = [st.container()]
        elif len(charts) == 2:
            cols = st.columns(2)
        else:
            cols = st.columns(3)

        for i, chart in enumerate(charts):
            col_index = i % len(cols)

            with cols[col_index]:
                try:
                    if chart.startswith('data:image'):
                        # Base64 image data
                        img_data = chart.split(',')[1]
                        st.image(
                            base64.b64decode(img_data),
                            caption=f"Chart {i+1}",
                            use_column_width=True
                        )
                    elif len(chart) > 100:  # Assume base64 data
                        # Raw base64 data
                        st.image(
                            base64.b64decode(chart),
                            caption=f"Chart {i+1}",
                            use_column_width=True
                        )
                    else:
                        # Chart identifier - get from session state
                        chart_data = st.session_state.get('chart_data', {})
                        if chart in chart_data:
                            st.image(
                                base64.b64decode(chart_data[chart]),
                                caption=chart.replace('_', ' ').title(),
                                use_column_width=True
                            )
                        else:
                            st.error(f"Chart '{chart}' not found")

                except Exception as e:
                    st.error(f"Error displaying chart: {e}")

    def _render_chat_input(self) -> None:
        """Render the chat input area."""
        # Chat input with dynamic key to force refresh
        if prompt := st.chat_input(
            "Ask me about your data...",
            key=f"chat_input_{st.session_state.chat_input_key}",
            disabled=st.session_state.processing_query
        ):
            self._handle_user_input(prompt)

        # Show processing indicator
        if st.session_state.processing_query:
            st.info("🤔 Processing your request...")

    def _render_chat_controls(self) -> None:
        """Render chat controls and options."""
        with st.expander("🔧 Chat Options", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🗑️ Clear Chat", type="secondary"):
                    self._clear_chat_history()

            with col2:
                show_metadata = st.checkbox(
                    "Show Metadata",
                    value=st.session_state.show_message_metadata,
                    help="Show detailed information about messages"
                )
                st.session_state.show_message_metadata = show_metadata

            with col3:
                auto_scroll = st.checkbox(
                    "Auto Scroll",
                    value=st.session_state.auto_scroll,
                    help="Automatically scroll to latest messages"
                )
                st.session_state.auto_scroll = auto_scroll

            # Quick action buttons when data is available
            if st.session_state.available_data:
                st.markdown("**Quick Actions:**")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if st.button("📊 Analyze Data", type="primary"):
                        self._handle_quick_action("Perform a comprehensive analysis of my data")

                with col2:
                    if st.button("📈 Show Patterns", type="secondary"):
                        self._handle_quick_action("Identify patterns and trends in my data")

                with col3:
                    if st.button("🔍 Find Outliers", type="secondary"):
                        self._handle_quick_action("Detect outliers in my data")

                with col4:
                    if st.button("📋 Generate Report", type="secondary"):
                        self._handle_quick_action("Generate a comprehensive report")

    def _handle_user_input(self, prompt: str) -> None:
        """
        Handle user input and generate appropriate response.

        Args:
            prompt: User's input message
        """
        # Add user message
        user_message = self._create_message(
            content=prompt,
            role=MessageType.USER
        )
        st.session_state.chat_messages.append(user_message)

        # Process the request
        st.session_state.processing_query = True

        # Refresh input by incrementing key
        st.session_state.chat_input_key += 1

        # Rerun to show user message immediately
        st.rerun()

    def _handle_quick_action(self, action_prompt: str) -> None:
        """
        Handle quick action button clicks.

        Args:
            action_prompt: The prompt to process
        """
        self._handle_user_input(action_prompt)

    def _process_user_query(self, query: str) -> None:
        """
        Process user query and generate appropriate response.

        Args:
            query: User's query to process
        """
        try:
            # Determine query type and generate response
            if self._is_analysis_request(query):
                response = self._handle_analysis_request(query)
            elif self._is_report_request(query):
                response = self._handle_report_request(query)
            elif self._is_help_request(query):
                response = self._handle_help_request(query)
            else:
                response = self._handle_general_query(query)

            # Add assistant response
            if response:
                assistant_message = self._create_message(
                    content=response.get('content', ''),
                    role=MessageType.ASSISTANT,
                    metadata=response.get('metadata', {}),
                    charts=response.get('charts', [])
                )
                st.session_state.chat_messages.append(assistant_message)

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_message = self._create_message(
                content=f"I encountered an error processing your request: {e}",
                role=MessageType.ERROR
            )
            st.session_state.chat_messages.append(error_message)

        finally:
            st.session_state.processing_query = False

    def _is_analysis_request(self, query: str) -> bool:
        """Check if query is requesting data analysis."""
        analysis_keywords = [
            'analyze', 'analysis', 'examine', 'investigate', 'study',
            'pattern', 'trend', 'correlation', 'outlier', 'distribution',
            'statistics', 'statistical', 'fraud', 'anomaly'
        ]
        return any(keyword in query.lower() for keyword in analysis_keywords)

    def _is_report_request(self, query: str) -> bool:
        """Check if query is requesting a report."""
        report_keywords = [
            'report', 'summary', 'document', 'generate', 'create',
            'write', 'markdown', 'export', 'download'
        ]
        return any(keyword in query.lower() for keyword in report_keywords)

    def _is_help_request(self, query: str) -> bool:
        """Check if query is requesting help."""
        help_keywords = ['help', 'how', 'what can', 'what do', 'explain', 'guide']
        return any(keyword in query.lower() for keyword in help_keywords)

    def _handle_analysis_request(self, query: str) -> Dict[str, Any]:
        """Handle analysis requests using the Analyser Agent."""
        if not st.session_state.available_data:
            return {
                'content': "I need data to perform analysis. Please upload a CSV file first.",
                'metadata': {'request_type': 'analysis', 'status': 'no_data'}
            }

        if not self.analyser_agent:
            return {
                'content': "Analysis agent is not available. Please configure an LLM provider first.",
                'metadata': {'request_type': 'analysis', 'status': 'no_agent'}
            }

        try:
            # Determine analysis type from query
            analysis_type = self._determine_analysis_type(query)

            # Create analysis request
            request = AnalysisRequest(
                statistics_data=st.session_state.statistics_data,
                pattern_data=st.session_state.pattern_data,
                chart_metadata=st.session_state.chart_data,
                analysis_type=analysis_type,
                dataset_name=st.session_state.get('processed_data', {}).get('file_info', {}).get('filename', 'dataset'),
                stream_callback=self._streaming_callback if self.enable_streaming else None
            )

            # Perform analysis
            if self.enable_streaming:
                result = self._perform_streaming_analysis(request)
            else:
                result = self.analyser_agent.analyze_data(request)

            # Store result
            st.session_state.last_analysis_result = result

            # Format response
            return self._format_analysis_response(result, query)

        except Exception as e:
            logger.error(f"Analysis request failed: {e}")
            return {
                'content': f"I encountered an error during analysis: {e}",
                'metadata': {'request_type': 'analysis', 'status': 'error', 'error': str(e)}
            }

    def _handle_report_request(self, query: str) -> Dict[str, Any]:
        """Handle report generation requests using the Reporter Agent."""
        if not st.session_state.available_data:
            return {
                'content': "I need analyzed data to generate a report. Please upload and analyze a CSV file first.",
                'metadata': {'request_type': 'report', 'status': 'no_data'}
            }

        if not self.reporter_agent:
            return {
                'content': "Report agent is not available. Please configure an LLM provider first.",
                'metadata': {'request_type': 'report', 'status': 'no_agent'}
            }

        # Check if we have analysis results
        analysis_results = st.session_state.get('last_analysis_result')
        if not analysis_results:
            # Perform analysis first
            analysis_response = self._handle_analysis_request("Analyze this data for reporting")
            if 'error' in analysis_response.get('metadata', {}):
                return analysis_response

            analysis_results = st.session_state.get('last_analysis_result')

        try:
            # Create report request
            request = ReportRequest(
                analysis_results=analysis_results.analysis_data,
                available_charts=list(st.session_state.chart_data.keys()) if st.session_state.chart_data else None,
                chart_data=st.session_state.chart_data,
                dataset_name=st.session_state.get('processed_data', {}).get('file_info', {}).get('filename', 'dataset'),
                analysis_type=analysis_results.analysis_type,
                report_style="comprehensive",
                stream_callback=self._streaming_callback if self.enable_streaming else None
            )

            # Generate report
            if self.enable_streaming:
                result = self._perform_streaming_report(request)
            else:
                result = self.reporter_agent.generate_report(request)

            # Store result
            st.session_state.last_report_result = result

            # Return the markdown report
            return {
                'content': result.markdown_report,
                'metadata': {
                    'request_type': 'report',
                    'status': 'success',
                    'processing_time': result.processing_time,
                    'report_style': result.report_style
                },
                'charts': list(st.session_state.chart_data.keys()) if st.session_state.chart_data else []
            }

        except Exception as e:
            logger.error(f"Report request failed: {e}")
            return {
                'content': f"I encountered an error generating the report: {e}",
                'metadata': {'request_type': 'report', 'status': 'error', 'error': str(e)}
            }

    def _handle_help_request(self, query: str) -> Dict[str, Any]:
        """Handle help and guidance requests."""
        help_content = """
# 🤖 AI Analytics Chatbot Help

## What I can do:

### 📊 Data Analysis
- **Comprehensive Analysis**: "Analyze my data" or "Perform a complete analysis"
- **Pattern Detection**: "Find patterns in my data" or "Identify trends"
- **Outlier Detection**: "Find outliers" or "Detect anomalies"
- **Correlation Analysis**: "Show correlations" or "Analyze relationships"
- **Fraud Analysis**: "Analyze fraud patterns" (for fraud detection datasets)

### 📈 Report Generation
- **Full Reports**: "Generate a report" or "Create a comprehensive report"
- **Summary Reports**: "Summarize my data" or "Give me a summary"
- **Technical Reports**: "Create a technical report"

### 🔍 Specific Questions
- "What is the fraud rate in my data?"
- "Which features are most important?"
- "Are there any temporal patterns?"
- "What's the data quality like?"

### 📋 Data Information
- "Describe my data"
- "What columns do I have?"
- "Show me basic statistics"

## Getting Started:
1. Upload a CSV file using the file uploader above
2. Wait for processing to complete
3. Ask me questions about your data
4. Use the quick action buttons for common tasks

## Tips:
- Be specific in your questions for better results
- I work best with fraud detection datasets (V1-V28, Time, Amount, Class)
- I can generate charts and visualizations
- Ask follow-up questions to dive deeper into analysis
        """

        return {
            'content': help_content,
            'metadata': {'request_type': 'help', 'status': 'success'}
        }

    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general conversational queries."""
        if not st.session_state.available_data:
            return {
                'content': "I'm ready to help analyze your data! Please upload a CSV file to get started. "
                          "Once you have data loaded, I can help you with analysis, pattern detection, "
                          "outlier identification, and report generation.",
                'metadata': {'request_type': 'conversation', 'status': 'no_data'}
            }

        # Provide contextual help based on available data
        file_info = st.session_state.get('processed_data', {}).get('file_info', {})
        filename = file_info.get('filename', 'your dataset')

        response = f"""I can see you have **{filename}** loaded and ready for analysis!

Here's what I can help you with:

**🔍 Quick Analysis Options:**
- "Analyze my data" - Comprehensive analysis
- "Find patterns" - Pattern and trend identification
- "Detect outliers" - Anomaly detection
- "Generate report" - Full analytical report

**📊 Data Overview:**
- Rows: {file_info.get('estimated_rows', 'Unknown'):,}
- Columns: {file_info.get('num_columns', 'Unknown')}
- Format: {'Fraud Detection' if file_info.get('has_fraud_columns') else 'General CSV'}

What would you like to explore first?"""

        return {
            'content': response,
            'metadata': {'request_type': 'conversation', 'status': 'data_available'}
        }

    def _determine_analysis_type(self, query: str) -> AnalysisType:
        """Determine the type of analysis requested from the query."""
        query_lower = query.lower()

        if any(word in query_lower for word in ['pattern', 'trend', 'temporal']):
            return AnalysisType.PATTERN_IDENTIFICATION
        elif any(word in query_lower for word in ['outlier', 'anomaly', 'unusual']):
            return AnalysisType.OUTLIER_DETECTION
        elif any(word in query_lower for word in ['correlation', 'relationship', 'feature']):
            return AnalysisType.RELATIONSHIP_ANALYSIS
        elif any(word in query_lower for word in ['fraud', 'class', 'prediction']):
            return AnalysisType.FRAUD_ANALYSIS
        elif any(word in query_lower for word in ['describe', 'overview', 'summary']):
            return AnalysisType.DATA_DESCRIPTION
        else:
            return AnalysisType.COMPREHENSIVE

    def _format_analysis_response(self, result: AnalysisResult, original_query: str) -> Dict[str, Any]:
        """Format analysis results for chat display."""
        analysis_data = result.analysis_data

        # Extract key insights
        summary = analysis_data.get('analysis_summary', {})
        findings = summary.get('key_findings', [])
        fraud_insights = summary.get('fraud_insights', '')

        # Create a conversational response
        response_parts = []

        response_parts.append(f"## 📊 Analysis Results\n")

        if summary.get('dataset_overview'):
            response_parts.append(f"**Dataset Overview:** {summary['dataset_overview']}\n")

        if findings:
            response_parts.append("**Key Findings:**")
            for finding in findings[:5]:  # Limit to 5 findings
                response_parts.append(f"• {finding}")
            response_parts.append("")

        if fraud_insights:
            response_parts.append(f"**Fraud Detection Insights:** {fraud_insights}\n")

        # Add confidence information
        confidence_scores = analysis_data.get('confidence_scores', {})
        if confidence_scores:
            response_parts.append("**Analysis Confidence:**")
            for metric, score in confidence_scores.items():
                if isinstance(score, (int, float)):
                    response_parts.append(f"• {metric.replace('_', ' ').title()}: {score:.2%}")

        # Add processing information
        response_parts.append(f"\n*Analysis completed in {result.processing_time:.2f} seconds*")

        return {
            'content': '\n'.join(response_parts),
            'metadata': {
                'request_type': 'analysis',
                'status': 'success',
                'analysis_type': result.analysis_type.value,
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time
            },
            'charts': []  # Charts will be handled separately if needed
        }

    def _streaming_callback(self, chunk: str) -> None:
        """Handle streaming chunks from agents."""
        if chunk:
            st.session_state.current_response += chunk
            # In a real implementation, this would update the UI in real-time

    def _perform_streaming_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform analysis with streaming (placeholder for actual streaming)."""
        # For now, just use regular analysis
        # In a full implementation, this would handle real-time streaming
        return self.analyser_agent.analyze_data(request)

    def _perform_streaming_report(self, request: ReportRequest) -> ReportResult:
        """Perform report generation with streaming (placeholder for actual streaming)."""
        # For now, just use regular report generation
        # In a full implementation, this would handle real-time streaming
        return self.reporter_agent.generate_report(request)

    def _create_message(self,
                       content: str,
                       role: MessageType,
                       metadata: Dict[str, Any] = None,
                       charts: List[str] = None) -> ChatMessage:
        """Create a new chat message."""
        st.session_state.message_counter += 1

        return ChatMessage(
            content=content,
            role=role,
            timestamp=datetime.now().isoformat(),
            message_id=f"msg_{st.session_state.message_counter}",
            metadata=metadata or {},
            charts=charts or []
        )

    def _add_system_message(self, content: str, show_suggestions: bool = False) -> None:
        """Add a system message to the chat."""
        if show_suggestions and not st.session_state.available_data:
            content += "\n\n**Suggested Actions:**\n"
            content += "• Upload a CSV file using the uploader above\n"
            content += "• Try the sample fraud detection dataset\n"
            content += "• Ask me 'help' for more information"

        message = self._create_message(content, MessageType.SYSTEM)
        st.session_state.chat_messages.append(message)

    def _clear_chat_history(self) -> None:
        """Clear the chat history."""
        st.session_state.chat_messages = []
        st.session_state.chat_history = []
        st.session_state.message_counter = 0
        st.session_state.current_response = ''
        st.rerun()

    def get_chat_history(self) -> List[ChatMessage]:
        """Get the current chat history."""
        return st.session_state.chat_messages.copy()

    def export_chat_history(self, format: str = "markdown") -> str:
        """
        Export chat history in specified format.

        Args:
            format: Export format ("markdown", "json", "text")

        Returns:
            Formatted chat history
        """
        if format == "markdown":
            return self._export_as_markdown()
        elif format == "json":
            return self._export_as_json()
        elif format == "text":
            return self._export_as_text()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_as_markdown(self) -> str:
        """Export chat history as markdown."""
        lines = ["# Chat History\n"]

        for message in st.session_state.chat_messages:
            role_icon = {
                MessageType.USER: "👤",
                MessageType.ASSISTANT: "🤖",
                MessageType.SYSTEM: "ℹ️",
                MessageType.ERROR: "⚠️"
            }.get(message.role, "•")

            lines.append(f"## {role_icon} {message.role.value.title()}")
            lines.append(f"*{message.timestamp}*\n")
            lines.append(message.content)
            lines.append("\n---\n")

        return "\n".join(lines)

    def _export_as_json(self) -> str:
        """Export chat history as JSON."""
        import json

        history_data = []
        for message in st.session_state.chat_messages:
            history_data.append({
                'role': message.role.value,
                'content': message.content,
                'timestamp': message.timestamp,
                'message_id': message.message_id,
                'metadata': message.metadata,
                'charts': message.charts
            })

        return json.dumps(history_data, indent=2)

    def _export_as_text(self) -> str:
        """Export chat history as plain text."""
        lines = ["Chat History\n" + "="*50 + "\n"]

        for message in st.session_state.chat_messages:
            lines.append(f"{message.role.value.upper()} ({message.timestamp}):")
            lines.append(message.content)
            lines.append("-" * 30)

        return "\n".join(lines)


# Convenience function for easy integration
def render_chat_interface(analyser_agent: Optional[AnalyserAgent] = None,
                         reporter_agent: Optional[ReporterAgent] = None,
                         enable_streaming: bool = True) -> None:
    """
    Convenience function to render chat interface.

    Args:
        analyser_agent: Analyser agent instance
        reporter_agent: Reporter agent instance
        enable_streaming: Whether to enable streaming responses
    """
    # Get agents from session state if not provided
    if not analyser_agent:
        analyser_agent = st.session_state.get('analyser_agent')

    if not reporter_agent:
        reporter_agent = st.session_state.get('reporter_agent')

    chat = ChatInterface(
        analyser_agent=analyser_agent,
        reporter_agent=reporter_agent,
        enable_streaming=enable_streaming
    )

    chat.render()

    # Process any pending queries
    if st.session_state.processing_query and st.session_state.chat_messages:
        last_message = st.session_state.chat_messages[-1]
        if last_message.role == MessageType.USER:
            chat._process_user_query(last_message.content)
            st.rerun()


if __name__ == "__main__":
    # Example usage for testing
    st.set_page_config(
        page_title="Chat Interface Test",
        page_icon="💬",
        layout="wide"
    )

    st.title("💬 Chat Interface Component Test")

    # Test the chat interface
    render_chat_interface()