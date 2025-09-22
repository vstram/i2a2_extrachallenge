"""
Streamlit AI Analytics Chatbot - Main Application

This is the main Streamlit application for the AI analytics chatbot that processes
CSV files and generates analytical reports using a dual-agent system with streaming
capabilities.
"""

import streamlit as st
import os
import sys
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

# Configure page settings
st.set_page_config(
    page_title="AI Analytics Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils.llm_config import LLMProvider, create_llm_manager, LLMManager
from agents.analyser import create_analyser_agent, AnalyserAgent
from agents.reporter import create_reporter_agent, ReporterAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionState:
    """Manage Streamlit session state with proper initialization."""

    @staticmethod
    def initialize():
        """Initialize all session state variables."""
        defaults = {
            # LLM Configuration
            'llm_provider': LLMProvider.OPENAI.value,
            'openai_api_key': '',
            'ollama_base_url': 'http://localhost:11434',
            'ollama_model': 'llama3.1:latest',

            # Application State
            'llm_manager': None,
            'analyser_agent': None,
            'reporter_agent': None,
            'agents_initialized': False,

            # Chat Interface
            'messages': [],
            'chat_history': [],

            # File Processing
            'uploaded_file': None,
            'processed_data': None,
            'statistics_data': None,
            'pattern_data': None,
            'chart_data': None,

            # UI State
            'show_debug': False,
            'processing_status': '',
            'last_error': None
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value


class LLMConfigurationSidebar:
    """Handle LLM configuration in the sidebar."""

    @staticmethod
    def render():
        """Render the LLM configuration sidebar."""
        st.sidebar.header("🤖 LLM Configuration")

        # Provider selection
        provider_options = {
            'OpenAI API': LLMProvider.OPENAI.value,
            'Ollama (Local)': LLMProvider.OLLAMA.value
        }

        selected_provider = st.sidebar.selectbox(
            "Select LLM Provider",
            options=list(provider_options.keys()),
            index=0 if st.session_state.llm_provider == LLMProvider.OPENAI.value else 1,
            help="Choose between OpenAI API (cloud) or Ollama (local)"
        )

        st.session_state.llm_provider = provider_options[selected_provider]

        # OpenAI Configuration
        if st.session_state.llm_provider == LLMProvider.OPENAI.value:
            LLMConfigurationSidebar._render_openai_config()

        # Ollama Configuration
        elif st.session_state.llm_provider == LLMProvider.OLLAMA.value:
            LLMConfigurationSidebar._render_ollama_config()

        # Connection test button
        if st.sidebar.button("🔗 Test Connection", type="secondary"):
            LLMConfigurationSidebar._test_connection()

        # Configuration status
        LLMConfigurationSidebar._show_connection_status()

    @staticmethod
    def _render_openai_config():
        """Render OpenAI specific configuration."""
        st.sidebar.subheader("OpenAI Settings")

        # API Key input
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Enter your OpenAI API key. You can get one from https://platform.openai.com/api-keys"
        )
        st.session_state.openai_api_key = api_key

        # Model selection
        model_options = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ]

        selected_model = st.sidebar.selectbox(
            "Model",
            options=model_options,
            index=0,
            help="Choose the OpenAI model to use"
        )
        st.session_state.openai_model = selected_model

        # Temperature setting
        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses (0.0 = deterministic, 1.0 = very creative)"
        )
        st.session_state.temperature = temperature

        # Show usage warning
        if not api_key:
            st.sidebar.warning("⚠️ Please enter your OpenAI API key to use the service.")
        else:
            st.sidebar.success("✅ API key configured")

    @staticmethod
    def _render_ollama_config():
        """Render Ollama specific configuration."""
        st.sidebar.subheader("Ollama Settings")

        # Base URL
        base_url = st.sidebar.text_input(
            "Ollama Base URL",
            value=st.session_state.ollama_base_url,
            help="URL where Ollama server is running (default: http://localhost:11434)"
        )
        st.session_state.ollama_base_url = base_url

        # Model selection
        model = st.sidebar.text_input(
            "Model Name",
            value=st.session_state.ollama_model,
            help="Name of the Ollama model to use (e.g., llama3.1:latest, gemma2:2b)"
        )
        st.session_state.ollama_model = model

        # Temperature setting
        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses"
        )
        st.session_state.temperature = temperature

        # Show connection info
        st.sidebar.info(
            "💡 Make sure Ollama is running locally:\n"
            "```bash\n"
            "ollama serve\n"
            "```"
        )

    @staticmethod
    def _test_connection():
        """Test the LLM connection."""
        try:
            with st.sidebar.status("Testing connection...", expanded=True) as status:
                # Create LLM manager with current settings
                if st.session_state.llm_provider == LLMProvider.OPENAI.value:
                    if not st.session_state.openai_api_key:
                        st.sidebar.error("❌ Please enter OpenAI API key")
                        return

                    llm_manager = create_llm_manager(
                        openai_api_key=st.session_state.openai_api_key,
                        auto_detect_available=False
                    )
                else:
                    llm_manager = create_llm_manager(
                        ollama_base_url=st.session_state.ollama_base_url,
                        auto_detect_available=False
                    )

                # Test basic functionality
                test_messages = [{"role": "user", "content": "Hello, this is a connection test."}]

                # Try to get provider info
                provider_info = llm_manager.get_provider_info()

                status.update(label="✅ Connection successful!", state="complete")
                st.sidebar.success(f"Connected to {provider_info.get('active_provider', 'unknown')}")

                # Update session state
                st.session_state.llm_manager = llm_manager
                st.session_state.last_error = None

        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {str(e)}")
            st.session_state.last_error = str(e)
            st.session_state.llm_manager = None

    @staticmethod
    def _show_connection_status():
        """Show current connection status."""
        st.sidebar.divider()

        if st.session_state.llm_manager:
            provider_info = st.session_state.llm_manager.get_provider_info()
            st.sidebar.success(f"🟢 Connected: {provider_info.get('active_provider', 'Unknown')}")
        elif st.session_state.last_error:
            st.sidebar.error(f"🔴 Connection Error: {st.session_state.last_error}")
        else:
            st.sidebar.warning("🟡 Not Connected")


class ApplicationSidebar:
    """Handle application-specific sidebar settings."""

    @staticmethod
    def render():
        """Render the application settings sidebar."""
        st.sidebar.divider()
        st.sidebar.header("⚙️ Application Settings")

        # Debug mode toggle
        debug_mode = st.sidebar.checkbox(
            "Debug Mode",
            value=st.session_state.show_debug,
            help="Show debug information and processing details"
        )
        st.session_state.show_debug = debug_mode

        # File processing settings
        st.sidebar.subheader("File Processing")

        max_file_size = st.sidebar.slider(
            "Max File Size (MB)",
            min_value=1,
            max_value=500,
            value=150,
            help="Maximum allowed CSV file size"
        )
        st.session_state.max_file_size_mb = max_file_size

        # Agent settings
        st.sidebar.subheader("Agent Configuration")

        enable_streaming = st.sidebar.checkbox(
            "Enable Streaming",
            value=True,
            help="Enable real-time streaming of agent responses"
        )
        st.session_state.enable_streaming = enable_streaming

        # Clear data button
        if st.sidebar.button("🗑️ Clear All Data", type="secondary"):
            ApplicationSidebar._clear_session_data()

    @staticmethod
    def _clear_session_data():
        """Clear session data."""
        keys_to_clear = [
            'messages', 'chat_history', 'uploaded_file', 'processed_data',
            'statistics_data', 'pattern_data', 'chart_data', 'processing_status'
        ]

        for key in keys_to_clear:
            if key in st.session_state:
                st.session_state[key] = [] if key in ['messages', 'chat_history'] else None

        st.sidebar.success("🗑️ All data cleared!")
        st.rerun()


class AgentManager:
    """Manage AI agents initialization and configuration."""

    @staticmethod
    def initialize_agents():
        """Initialize the AI agents if LLM is configured."""
        if not st.session_state.llm_manager:
            return False

        try:
            if not st.session_state.agents_initialized:
                # Create agents with the configured LLM manager
                st.session_state.analyser_agent = AnalyserAgent(
                    llm_manager=st.session_state.llm_manager
                )
                st.session_state.reporter_agent = ReporterAgent(
                    llm_manager=st.session_state.llm_manager
                )

                st.session_state.agents_initialized = True
                logger.info("AI agents initialized successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            st.session_state.last_error = f"Agent initialization failed: {e}"
            return False

    @staticmethod
    def get_agent_status():
        """Get status of all agents."""
        if not st.session_state.agents_initialized:
            return {"status": "not_initialized"}

        try:
            analyser_status = st.session_state.analyser_agent.get_agent_status()
            reporter_status = st.session_state.reporter_agent.get_agent_status()

            return {
                "status": "ready",
                "analyser": analyser_status,
                "reporter": reporter_status
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


class MainInterface:
    """Main application interface."""

    @staticmethod
    def render():
        """Render the main application interface."""
        # Application header
        st.title("🤖 AI Analytics Chatbot")
        st.markdown(
            "**Process CSV files and generate analytical reports using AI agents**"
        )

        # Show connection status
        MainInterface._show_connection_status()

        # Main content area
        if st.session_state.llm_manager and AgentManager.initialize_agents():
            MainInterface._render_main_content()
        else:
            MainInterface._render_setup_instructions()

    @staticmethod
    def _show_connection_status():
        """Show connection and agent status."""
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.session_state.llm_manager:
                st.success("🟢 LLM Connected")
            else:
                st.error("🔴 LLM Not Connected")

        with col2:
            if st.session_state.agents_initialized:
                st.success("🤖 Agents Ready")
            else:
                st.warning("⚠️ Agents Not Ready")

        with col3:
            if st.session_state.uploaded_file:
                st.info(f"📁 {st.session_state.uploaded_file.name}")
            else:
                st.info("📁 No File Uploaded")

    @staticmethod
    def _render_setup_instructions():
        """Render setup instructions when not ready."""
        st.info("🔧 **Setup Required**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 1. Configure LLM Provider

            **Option A: OpenAI API**
            - Get an API key from [OpenAI](https://platform.openai.com/api-keys)
            - Enter the API key in the sidebar
            - Test the connection

            **Option B: Ollama (Local)**
            - Install Ollama locally
            - Start the Ollama server
            - Configure the model in the sidebar
            """)

        with col2:
            st.markdown("""
            ### 2. Upload CSV File

            - Prepare your CSV file (fraud detection format preferred)
            - Use the file uploader in the main interface
            - Wait for automatic processing

            ### 3. Start Analyzing

            - Ask questions about your data
            - Request specific analysis types
            - Generate comprehensive reports
            """)

        # Show agent status for debugging
        if st.session_state.show_debug:
            st.subheader("🔍 Debug Information")
            agent_status = AgentManager.get_agent_status()
            st.json(agent_status)

    @staticmethod
    def _render_main_content():
        """Render the main content when everything is ready."""
        # File upload section
        st.subheader("📁 Data Upload")

        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help=f"Upload CSV files up to {st.session_state.get('max_file_size_mb', 150)}MB"
        )

        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")

        # Chat interface placeholder
        st.subheader("💬 Chat Interface")

        # Placeholder for chat messages
        chat_container = st.container()

        with chat_container:
            if st.session_state.messages:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            else:
                st.info("👋 Welcome! Upload a CSV file and ask me questions about your data.")

        # Chat input
        if prompt := st.chat_input("Ask me about your data..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate assistant response (placeholder)
            with st.chat_message("assistant"):
                if not st.session_state.uploaded_file:
                    response = "Please upload a CSV file first so I can analyze your data."
                else:
                    response = f"I'll analyze your question: '{prompt}'. This functionality will be implemented in the next tasks (file processing and chat interface components)."

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

        # Processing status
        if st.session_state.processing_status:
            st.info(f"⚙️ {st.session_state.processing_status}")

        # Debug information
        if st.session_state.show_debug:
            MainInterface._render_debug_info()

    @staticmethod
    def _render_debug_info():
        """Render debug information."""
        st.subheader("🔍 Debug Information")

        with st.expander("Session State", expanded=False):
            debug_state = {
                k: v for k, v in st.session_state.items()
                if k not in ['llm_manager', 'analyser_agent', 'reporter_agent']  # Exclude complex objects
            }
            st.json(debug_state)

        with st.expander("Agent Status", expanded=False):
            agent_status = AgentManager.get_agent_status()
            st.json(agent_status)

        with st.expander("LLM Configuration", expanded=False):
            if st.session_state.llm_manager:
                llm_info = st.session_state.llm_manager.get_provider_info()
                st.json(llm_info)
            else:
                st.write("No LLM manager configured")


def main():
    """Main application entry point."""
    try:
        # Initialize session state
        SessionState.initialize()

        # Render sidebar
        with st.sidebar:
            LLMConfigurationSidebar.render()
            ApplicationSidebar.render()

        # Render main interface
        MainInterface.render()

        # Footer
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.caption("🤖 AI Analytics Chatbot")

        with col2:
            st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        with col3:
            st.caption("🔧 Built with Streamlit")

    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        logger.error(f"Application error: {e}")

        if st.session_state.show_debug:
            st.exception(e)


if __name__ == "__main__":
    main()