# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based AI analytics chatbot that processes CSV files and generates analytical reports using a dual-agent system with streaming capabilities.

## Architecture

The application uses a **dual AI agent workflow**:

1. **CSV Processing Pipeline**: Large CSV files are pre-processed into statistical JSON summaries to bypass LLM token limitations
2. **Analyser Agent** (LangChain): Processes the JSON statistics and extracts insights
3. **Report Agent** (LangChain): Converts analysis results into formatted markdown reports
4. **Streaming UI**: Real-time display of results in Streamlit chat interface

### Key Components

- `app.py` - Main Streamlit application with chat UI and file upload
- `agents/analyser.py` - Analyser Agent implementation using LangChain
- `agents/reporter.py` - Report Agent for markdown generation
- `utils/csv_processor.py` - CSV to JSON statistics converter
- `utils/llm_config.py` - LLM configuration for OpenAI/Ollama integration

## Development Commands

### Running the Application
```bash
streamlit run app.py
```

### Installing Dependencies
```bash
pip install streamlit langchain openai pandas

# For Ollama support
pip install langchain-ollama
```

### Environment Setup
```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# For Ollama (install and pull model)
ollama pull llama2
```

## Technical Requirements

The application must support comprehensive data analysis capabilities including:

**Data Description Analysis**:
- Data type identification (numeric, categorical)
- Distribution analysis (histograms, distributions)
- Range analysis (min/max values)
- Central tendency measures (mean, median)
- Variability metrics (standard deviation, variance)

**Pattern Recognition**:
- Temporal patterns and trends
- Frequency analysis
- Data clustering

**Outlier Detection**:
- Outlier identification
- Impact assessment
- Data cleaning recommendations

**Relationship Analysis**:
- Variable correlations
- Cross-variable influence
- Scatter plot and crosstab insights

## LLM Integration

The system supports dual LLM backends:
- **OpenAI API**: Cloud-based processing with API key authentication
- **Ollama**: Local LLM execution for privacy/offline usage

Both backends integrate through LangChain for consistent agent behavior and streaming support.

## Test Data

The repository includes test CSV files in the `data/` directory for development and testing:

- **`creditcard-huge.csv`** (~150MB): Large-scale fraud detection dataset for performance testing
- **`creditcard-tiny.csv`** (~120KB): Smaller subset for rapid development and debugging

Both files contain fraud detection data with:
- **V1-V28**: PCA-transformed features
- **Time**: Seconds elapsed from first transaction
- **Amount**: Transaction amount
- **Class**: Fraud indicator (0=normal, 1=fraud)

## Testing

### Running Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_csv_processor.py

# Run with verbose output
python -m pytest -v tests/
```

### Test Development Guidelines
- Create unit tests for all utility functions and agents
- Use the small CSV file (`creditcard-tiny.csv`) for test fixtures
- Mock LLM calls to avoid API costs during testing
- Test edge cases like empty files, invalid formats, and network failures
- Ensure tests can run independently without the full Streamlit application