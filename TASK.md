# TASK.md - Development Tasks for Streamlit AI Analytics Chatbot

## Project Overview
Develop a Streamlit chatbot application that processes large CSV files (150MB+) and generates analytical reports using a dual-agent system. The application handles fraud detection data with PCA-transformed features V1-V28, Time, Amount, and Class variables.

## Task Breakdown

### Phase 1: Data Processing Infrastructure

#### Task 1.1: CSV Statistics Processor
**File**: `utils/csv_processor.py`
- Create function to load large CSV files efficiently (chunked reading)
- Generate comprehensive statistics for all columns:
  - Data types identification (numeric, categorical)
  - Basic statistics (count, mean, median, std, min, max, quartiles)
  - Distribution analysis (histograms, value counts)
  - Missing values analysis
- Handle fraud detection specific features:
  - Time series analysis for 'Time' column
  - Transaction amount distribution for 'Amount' column
  - Class distribution for fraud detection (0/1 ratio)
- Export statistics to JSON format

#### Task 1.2: Visualization Generator
**File**: `utils/chart_generator.py`
- Create histogram plots for numerical variables (V1-V28, Time, Amount)
- Generate correlation heatmap for all variables
- Create scatter plots for key variable relationships
- Generate fraud class distribution charts
- Encode all charts as base64 strings for JSON embedding
- Optimize chart generation for large datasets (sampling if needed)

#### Task 1.3: Pattern Analysis Module
**File**: `utils/pattern_analyzer.py`
- Temporal pattern detection for 'Time' variable
- Outlier detection using statistical methods (IQR, Z-score)
- Clustering analysis (K-means on PCA components)
- Correlation analysis between variables
- Fraud pattern identification (Class vs other variables)
- Export findings to structured JSON format

### Phase 2: LLM Configuration

#### Task 2.1: LLM Configuration Manager
**File**: `utils/llm_config.py`
- Implement OpenAI API integration with streaming support
- Implement Ollama local integration with streaming support
- Create configuration switcher (OpenAI vs Ollama)
- Handle API key management and validation
- Implement error handling and fallback mechanisms

#### Task 2.2: Prompt Templates
**File**: `utils/prompts.py`
- Create structured prompts for Analyser Agent
- Create structured prompts for Report Agent
- Include fraud detection domain knowledge in prompts
- Define JSON schema expectations for agent outputs
- Create prompt templates for different analysis types

### Phase 3: AI Agent System

#### Task 3.1: Analyser Agent
**File**: `agents/analyser.py`
- Implement LangChain agent to process statistics JSON
- Answer technical questions about data:
  - Data type analysis and distribution insights
  - Pattern and trend identification
  - Outlier detection and impact assessment
  - Variable relationship analysis
- Generate structured analysis JSON with conclusions
- Handle streaming responses for real-time feedback

#### Task 3.2: Report Agent
**File**: `agents/reporter.py`
- Implement LangChain agent to convert analysis to markdown
- Create formatted reports with:
  - Executive summary
  - Detailed findings sections
  - Embedded charts (base64 decoded)
  - Tables with key statistics
  - Fraud detection insights
- Generate markdown optimized for Streamlit display
- Support streaming report generation

### Phase 4: Streamlit UI

#### Task 4.1: Main Application Structure
**File**: `app.py`
- Create Streamlit app with sidebar for configuration
- Implement LLM provider selection (OpenAI/Ollama)
- Add API key input for OpenAI
- Create main chat interface layout
- Handle session state management

#### Task 4.2: File Upload Interface
**File**: `components/file_uploader.py`
- Implement CSV file upload with validation
- Add file size checks and warnings for large files
- Display upload progress and file information
- Trigger automatic statistics processing on upload
- Cache processed results to avoid reprocessing

#### Task 4.3: Chat Interface
**File**: `components/chat_interface.py`
- Create chat UI with message history
- Implement streaming message display
- Add support for markdown rendering
- Include image display for charts
- Handle user queries and agent responses

#### Task 4.4: Statistics Display
**File**: `components/stats_display.py`
- Create expandable sections for raw statistics
- Display data overview (shape, types, missing values)
- Show basic statistics tables
- Include interactive charts when possible
- Add download options for generated reports

### Phase 5: Integration and Testing

#### Task 5.1: Agent Workflow Integration
**File**: `workflows/analysis_workflow.py`
- Orchestrate CSV processing → Analyser Agent → Report Agent flow
- Handle error propagation and recovery
- Implement progress tracking and user feedback
- Manage data flow between components
- Add logging for debugging

#### Task 5.2: Error Handling
**File**: `utils/error_handler.py`
- Implement comprehensive error handling for:
  - Large file processing failures
  - LLM API failures and rate limits
  - Invalid CSV format handling
  - Network connectivity issues
- Create user-friendly error messages
- Add retry mechanisms where appropriate

#### Task 5.3: Performance Optimization
**File**: `utils/performance.py`
- Implement caching for processed statistics
- Add progress bars for long operations
- Optimize memory usage for large files
- Implement chunked processing where needed
- Add performance monitoring

### Phase 6: Configuration and Documentation

#### Task 6.1: Environment Configuration
**File**: `config/settings.py`
- Create configuration management system
- Handle environment variables
- Set default parameters for processing
- Configure logging levels
- Add development/production settings

#### Task 6.2: Requirements and Dependencies
**File**: `requirements.txt`
- List all required Python packages
- Pin versions for stability
- Include optional dependencies (Ollama support)
- Add development dependencies if needed

#### Task 6.3: Usage Documentation
**File**: Update `README.md`
- Add detailed setup instructions
- Include example usage scenarios
- Document fraud detection specific features
- Add troubleshooting section
- Include performance considerations

## Implementation Priority

1. **Core Data Processing** (Tasks 1.1-1.3): Essential for handling large CSV files
2. **LLM Integration** (Tasks 2.1-2.2): Required for agent functionality
3. **Agent Development** (Tasks 3.1-3.2): Core AI functionality
4. **Basic UI** (Tasks 4.1-4.3): User interface for interaction
5. **Integration** (Tasks 5.1-5.3): Connect all components
6. **Polish** (Tasks 4.4, 6.1-6.3): Enhanced features and documentation

## Key Technical Considerations

- **Memory Management**: Use chunked reading for 150MB+ CSV files
- **Processing Time**: Implement progress indicators for long operations
- **Fraud Detection Domain**: Leverage knowledge of financial transaction patterns
- **LLM Token Limits**: Keep JSON summaries concise but comprehensive
- **Streaming Support**: Ensure real-time feedback in chat interface
- **Error Recovery**: Graceful handling of processing failures