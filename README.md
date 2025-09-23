# Streamlit AI Analytics Chatbot

A powerful Streamlit-based chatbot application designed for fraud detection analysis that processes large CSV files (150MB+) and generates comprehensive analytical reports using a sophisticated dual-agent AI system with streaming capabilities.

## Overview

This application provides an intelligent data analysis workflow through a conversational interface specifically optimized for fraud detection datasets. It handles large CSV files by first generating statistical summaries, then using specialized AI agents to analyze the data and generate comprehensive, actionable reports.

### Key Capabilities

- **Large File Processing**: Efficiently handles CSV files up to 150MB+ using chunked processing and memory optimization
- **Fraud Detection Specialization**: Tailored for financial transaction analysis with PCA-transformed features
- **Intelligent Workflow**: Dual-agent system that bypasses LLM token limitations through statistical preprocessing
- **Real-time Analysis**: Interactive chat interface with streaming responses for immediate insights

## Features

### Core Features

- **Interactive Chat UI**: Streamlit-based chatbot interface with real-time streaming responses
- **Large CSV File Support**: Upload and process files up to 150MB+ with progress tracking
- **Dual AI Agent System**:
  - **Analyser Agent**: Processes JSON statistics from CSV files, performs deep statistical analysis
  - **Report Agent**: Generates formatted markdown reports with charts and insights
- **Advanced Analytics**:
  - Pattern recognition and trend analysis
  - Outlier detection and impact assessment
  - Correlation analysis between variables
  - Fraud-specific pattern identification
- **Flexible LLM Integration**:
  - OpenAI API support (GPT-3.5/GPT-4) with streaming
  - Local Ollama integration for privacy-focused deployments
- **Performance Optimization**:
  - Advanced caching system with LRU eviction
  - Memory optimization achieving up to 90% reduction
  - Chunked processing for large datasets
  - Real-time performance monitoring

## Architecture

1. **CSV Processing**: Large CSV files are pre-processed into statistical JSON summaries
2. **Analyser Agent**: Uses LangChain to analyze the JSON statistics
3. **Report Agent**: Converts analysis into formatted markdown reports
4. **UI Display**: Renders the markdown report in the Streamlit chat interface

## Technology Stack

- **Frontend**: Streamlit
- **AI Framework**: LangChain
- **LLM Options**:
  - OpenAI API
  - Ollama (local)
- **Data Processing**: Python/Pandas
- **UI Components**: File uploader, chat interface

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git (for development installation)

### Installation Options

#### Option 1: Basic Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/vstram/i2a2_extrachallenge.git
cd i2a2_extrachallenge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

#### Option 2: Feature-Specific Installation

```bash
# Full installation with all features
pip install -e .[all]

# Local LLM support only
pip install -e .[ollama]

# Performance monitoring
pip install -e .[performance]

# Development tools
pip install -e .[dev]

# Production deployment
pip install -e .[production]
```

#### Option 3: Docker Installation

```bash
# Using Docker Compose (recommended)
docker-compose up --build

# Manual Docker build
docker build -t ai-analytics-chatbot .
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key ai-analytics-chatbot
```

For detailed installation instructions including platform-specific setup, see [INSTALL.md](INSTALL.md).

## Configuration

### Environment Setup

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# LLM Configuration
OPENAI_API_KEY=your-openai-api-key-here
LLM_PROVIDER=openai  # or 'ollama'

# Application Settings
ENVIRONMENT=development  # development, testing, staging, production
LOG_LEVEL=INFO
CACHE_ENABLED=true
CACHE_MAX_SIZE_MB=1000

# Performance Settings
ENABLE_MONITORING=true
CHUNK_SIZE=10000
MAX_FILE_SIZE_MB=150
```

### OpenAI Setup

1. **Get API Key**: Obtain from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Set Environment Variable**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
3. **Configure in Application**: The app will automatically detect the API key

### Ollama Setup (Local LLM)

1. **Install Ollama**:
   ```bash
   # Linux/macOS
   curl -fsSL https://ollama.ai/install.sh | sh

   # Windows: Download from https://ollama.ai/download
   ```

2. **Pull Models**:
   ```bash
   # Recommended models
   ollama pull llama2       # 7B parameters, fast
   ollama pull llama2:13b   # 13B parameters, better quality
   ollama pull codellama    # Code-specialized model
   
   # To inspect the Ollama logs on mac
   tail -f ~/.ollama/logs/server.log   
   ```

3. **Start Ollama Service**:
   ```bash
   ollama serve
   ```

4. **Configure Environment**:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama2
   OLLAMA_BASE_URL=http://localhost:11434
   ```

## Usage

### Starting the Application

1. **Activate Virtual Environment**:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Start Streamlit**:
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**: Open http://localhost:8501 in your browser

### Using the Fraud Detection Analytics

#### Step 1: Configure LLM Provider

1. **Select Provider**: Choose between OpenAI or Ollama in the sidebar
2. **API Key Setup**:
   - For OpenAI: Enter your API key in the sidebar input
   - For Ollama: Ensure service is running locally

#### Step 2: Upload CSV File

1. **File Upload**: Use the file uploader in the sidebar
2. **Supported Formats**: CSV files up to 150MB
3. **File Validation**: System validates file format and size
4. **Processing**: Automatic statistical preprocessing begins

#### Step 3: Interactive Analysis

Ask questions about your fraud detection data:

**Data Exploration:**
```
"What is the overall structure and quality of this dataset?"
"Show me the distribution of transaction amounts"
"What percentage of transactions are fraudulent?"
```

**Pattern Analysis:**
```
"Are there temporal patterns in fraudulent transactions?"
"Which PCA components show the strongest correlation with fraud?"
"Identify outliers in the transaction amounts"
```

**Advanced Analysis:**
```
"Perform cluster analysis on the PCA components"
"What are the key indicators of fraudulent transactions?"
"Generate a comprehensive fraud detection report"
```

#### Step 4: Interpret Results

- **Real-time Streaming**: Watch analysis unfold in real-time
- **Interactive Charts**: Examine generated visualizations
- **Markdown Reports**: Read formatted analysis reports
- **Export Options**: Download results for further use

## System Workflow

### Overview

The application uses a sophisticated multi-stage pipeline optimized for large-scale fraud detection analysis:

```
CSV Upload → Statistical Processing → Analyser Agent → Report Agent → Streaming Display
```

### Detailed Workflow

1. **File Upload & Validation**
   - User uploads CSV file via Streamlit interface
   - System validates file format, size (max 150MB), and structure
   - Progress indicators show upload status

2. **Statistical Preprocessing**
   - **Chunked Reading**: Large files processed in memory-efficient chunks
   - **Data Analysis**: Comprehensive statistical analysis including:
     - Data types and distributions
     - Correlation matrices
     - Outlier detection
     - Temporal pattern analysis
   - **JSON Generation**: Statistics exported to structured JSON format
   - **Caching**: Results cached for immediate reuse

3. **Analyser Agent Processing**
   - **LangChain Integration**: Processes JSON statistics using configured LLM
   - **Domain Knowledge**: Applies fraud detection expertise
   - **Analysis Types**:
     - Descriptive statistics analysis
     - Pattern recognition and trend identification
     - Anomaly and outlier detection
     - Variable relationship analysis
   - **Streaming Output**: Real-time analysis results

4. **Report Agent Generation**
   - **Markdown Conversion**: Transforms analysis into formatted reports
   - **Chart Integration**: Embeds generated visualizations
   - **Structured Output**: Executive summaries and detailed findings
   - **Fraud-Specific Insights**: Tailored recommendations for fraud detection

5. **Interactive Display**
   - **Real-time Streaming**: Live updates as analysis progresses
   - **Chat Interface**: Conversational interaction with results
   - **Export Options**: Download reports and charts

### Performance Optimization

- **Memory Management**: Up to 90% memory reduction through optimization
- **Intelligent Caching**: LRU eviction with TTL for frequently accessed data
- **Progress Tracking**: Real-time feedback on long-running operations
- **Error Recovery**: Automatic retry mechanisms and graceful degradation

## Fraud Detection Capabilities

### Core Analysis Features

The application provides comprehensive fraud detection analysis capabilities:

#### 1. Data Description Analysis
- **Data Type Identification**: Automatic detection of numeric vs categorical variables
- **Distribution Analysis**: Generate histograms and statistical distributions for all features
- **Range Analysis**: Calculate minimum, maximum, and quartile values
- **Central Tendency**: Compute mean, median, and mode for all variables
- **Variability Metrics**: Standard deviation, variance, and coefficient of variation
- **Missing Value Analysis**: Identify and quantify missing data patterns

#### 2. Fraud-Specific Pattern Recognition
- **Temporal Fraud Patterns**: Analyze transaction timing to identify suspicious periods
- **Amount Distribution Analysis**: Detect unusual transaction amount patterns
- **PCA Component Analysis**: Identify which principal components correlate with fraud
- **Class Imbalance Assessment**: Analyze fraud vs legitimate transaction ratios
- **Frequency Analysis**: Identify most/least frequent patterns in fraud cases

#### 3. Advanced Outlier Detection
- **Statistical Methods**: IQR, Z-score, and modified Z-score outlier detection
- **Fraud Impact Assessment**: Evaluate how outliers affect fraud detection accuracy
- **Anomaly Clustering**: Group similar anomalous transactions
- **Recommendation Engine**: Suggest data cleaning and preprocessing strategies

#### 4. Variable Relationship Analysis
- **Correlation Matrices**: Comprehensive correlation analysis between all variables
- **Scatter Plot Analysis**: Visual relationship exploration between key variables
- **Cross-Tabulation**: Categorical variable relationship analysis
- **Feature Importance**: Identify variables with highest fraud prediction power
- **Interaction Effects**: Detect combined variable effects on fraud likelihood

#### 5. Fraud Detection Insights
- **Risk Scoring**: Generate fraud risk scores for transaction patterns
- **Decision Boundaries**: Identify optimal thresholds for fraud classification
- **Feature Engineering Recommendations**: Suggest new features for improved detection
- **Model Performance Indicators**: Assess potential classification accuracy

### Example Analysis Questions

**Basic Data Exploration:**
```
"What is the overall quality and structure of this fraud detection dataset?"
"How many transactions are in the dataset and what's the fraud rate?"
"What are the data types and distributions of all variables?"
```

**Fraud Pattern Analysis:**
```
"Are there specific time periods with higher fraud rates?"
"What transaction amounts are most commonly associated with fraud?"
"Which PCA components show the strongest correlation with fraudulent activity?"
```

**Advanced Fraud Detection:**
```
"Identify unusual patterns that might indicate new fraud schemes"
"What are the key distinguishing features of fraudulent transactions?"
"Generate a comprehensive fraud risk assessment report"
```

## Dataset Information

### Credit Card Fraud Detection Dataset

This application is optimized for credit card fraud detection datasets with the following structure:

#### Dataset Features

**PCA-Transformed Features (V1-V28):**
- **Type**: Numerical principal components
- **Source**: Result of PCA transformation on original confidential features
- **Usage**: Core features for fraud pattern detection
- **Confidentiality**: Original features masked for privacy protection

**Non-Transformed Features:**
- **Time**: Seconds elapsed between each transaction and the first transaction
  - **Type**: Numerical (continuous)
  - **Range**: 0 to maximum transaction time
  - **Usage**: Temporal pattern analysis, fraud timing detection

- **Amount**: Transaction amount in monetary units
  - **Type**: Numerical (continuous)
  - **Range**: Varies by dataset (typically 0 to several thousand)
  - **Usage**: Amount-based fraud detection, cost-sensitive learning

- **Class**: Fraud indicator (target variable)
  - **Type**: Binary categorical (0/1)
  - **Values**: 0 = Legitimate transaction, 1 = Fraudulent transaction
  - **Usage**: Ground truth for fraud detection validation

#### Dataset Characteristics

- **Size**: Typically 150MB+ for comprehensive fraud detection
- **Class Imbalance**: Fraud cases usually represent <1% of transactions
- **Temporal Span**: Often covers multiple days/weeks of transaction data
- **Privacy**: PCA transformation ensures anonymization while preserving patterns

#### Test Data

The repository includes two test datasets in the `data/` directory:

1. **`creditcard-huge.csv`** (~150MB)
   - Large-scale dataset for performance testing
   - Full fraud detection feature set
   - Suitable for production-scale analysis

2. **`creditcard-tiny.csv`** (~120KB)
   - Smaller subset for rapid development
   - Same feature structure as large dataset
   - Ideal for testing and debugging

## Project Structure

```
i2a2_extrachallenge/
├── app.py                          # Main Streamlit application
├── agents/                         # AI Agent System
│   ├── __init__.py
│   ├── analyser.py                 # Analyser Agent (LangChain-based)
│   └── reporter.py                 # Report Agent (Markdown generation)
├── components/                     # Streamlit UI Components
│   ├── __init__.py
│   ├── chat_interface.py           # Chat UI with streaming
│   ├── file_uploader.py            # CSV file upload interface
│   └── stats_display.py            # Statistics display component
├── config/                         # Configuration Management
│   ├── __init__.py
│   └── settings.py                 # Environment-aware configuration
├── utils/                          # Core Utilities
│   ├── __init__.py
│   ├── csv_processor.py            # CSV to JSON statistics converter
│   ├── chart_generator.py          # Visualization generation
│   ├── pattern_analyzer.py         # Pattern detection and analysis
│   ├── llm_config.py               # LLM provider configuration
│   ├── prompts.py                  # AI agent prompt templates
│   ├── error_handler.py            # Comprehensive error handling
│   └── performance.py              # Performance optimization utilities
├── workflows/                      # Analysis Orchestration
│   ├── __init__.py
│   └── analysis_workflow.py        # End-to-end workflow management
├── data/                           # Test Datasets
│   ├── creditcard-huge.csv         # Large-scale test data (~150MB)
│   ├── creditcard-tiny.csv         # Development test data (~120KB)
│   └── charts_output/              # Generated visualization cache
├── cache/                          # Performance Caching
├── logs/                           # Application Logging
├── requirements.txt                # Core dependencies
├── requirements-dev.txt            # Development dependencies
├── requirements-optional.txt       # Optional feature dependencies
├── requirements-production.txt     # Production deployment dependencies
├── pyproject.toml                  # Modern Python packaging configuration
├── .env.example                    # Environment configuration template
├── INSTALL.md                      # Detailed installation instructions
├── CLAUDE.md                       # Development guidance
└── README.md                       # This file
```

### Key Components

**Core Application:**
- `app.py`: Main Streamlit interface with sidebar configuration and chat UI
- `components/`: Modular UI components for file upload, chat, and statistics display

**AI Processing:**
- `agents/`: LangChain-based AI agents for analysis and reporting
- `workflows/`: Orchestration of the complete analysis pipeline

**Data Processing:**
- `utils/csv_processor.py`: Efficient large file processing with chunking
- `utils/pattern_analyzer.py`: Advanced statistical analysis and pattern detection
- `utils/chart_generator.py`: Dynamic visualization generation

**System Infrastructure:**
- `config/settings.py`: Environment-aware configuration management
- `utils/performance.py`: Memory optimization and caching systems
- `utils/error_handler.py`: Comprehensive error handling and recovery

## System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **RAM**: 4GB (8GB recommended for large datasets)
- **Storage**: 2GB free space for caching and temporary files
- **CPU**: 2 cores (4+ cores recommended for optimal performance)
- **Network**: Internet connection for OpenAI API (if using)

### Recommended Requirements
- **RAM**: 16GB for processing 150MB+ datasets efficiently
- **Storage**: 10GB for extensive caching and logging
- **CPU**: 8+ cores for parallel processing optimization
- **SSD**: For faster I/O operations with large files

### Software Dependencies

**Core Dependencies:**
- Python 3.11+
- Streamlit ≥1.49.1
- LangChain ≥0.3.27
- Pandas ≥2.3.2
- NumPy ≥2.3.0
- Scikit-learn ≥1.7.0

**Optional Dependencies:**
- OpenAI API (for cloud LLM)
- Ollama (for local LLM)
- Performance monitoring tools (psutil, memory-profiler)
- Advanced visualization libraries (plotly-dash, bokeh)

## Performance Considerations

### Large File Processing

**Memory Management:**
- Files >50MB automatically use chunked processing
- Memory optimization reduces usage by up to 90%
- Intelligent caching prevents reprocessing

**Processing Time:**
- 150MB file: ~2-5 minutes for initial processing
- Subsequent analysis: <30 seconds (cached)
- Real-time streaming provides immediate feedback

**Optimization Strategies:**
```bash
# Enable performance monitoring
export ENABLE_MONITORING=true

# Increase cache size for better performance
export CACHE_MAX_SIZE_MB=2000

# Optimize chunk size for your system
export CHUNK_SIZE=50000
```

### LLM Performance

**OpenAI API:**
- Streaming responses for real-time feedback
- Rate limiting handled automatically
- Retry mechanisms for transient failures

**Ollama Local:**
- Faster response times (no network latency)
- Higher memory usage (model loading)
- Privacy benefits (no data leaves local machine)

**Performance Tips:**
- Use GPT-3.5-turbo for faster responses
- Use GPT-4 for higher quality analysis
- Local Ollama: Use llama2:7b for speed, llama2:13b for quality

## Troubleshooting

### Common Issues

#### 1. Installation Problems

**Issue**: `ModuleNotFoundError` for core packages
```bash
# Solution: Ensure virtual environment is activated
source .venv/bin/activate
pip install -r requirements.txt
```

**Issue**: Permission errors during installation
```bash
# Solution: Use virtual environment instead of system Python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2. File Upload Issues

**Issue**: "File too large" error
- **Cause**: File exceeds 150MB limit
- **Solution**: Use smaller dataset or increase `MAX_FILE_SIZE_MB` in config

**Issue**: "Invalid CSV format" error
- **Cause**: File is not properly formatted CSV
- **Solution**: Ensure file has headers and is comma-separated

**Issue**: Upload hangs or fails
```bash
# Check available disk space
df -h

# Verify file permissions
ls -la your-file.csv
```

#### 3. LLM Configuration Issues

**OpenAI API Problems:**
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API connectivity
python -c "import openai; print('API key valid')"
```

**Ollama Connection Issues:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama service
ollama serve

# Verify model is available
ollama list
```

#### 4. Performance Issues

**High Memory Usage:**
```bash
# Enable memory optimization
export ENABLE_MEMORY_OPTIMIZATION=true

# Reduce chunk size
export CHUNK_SIZE=5000

# Clear cache
rm -rf cache/*
```

**Slow Processing:**
```bash
# Check system resources
top
df -h

# Enable performance monitoring
export ENABLE_MONITORING=true

# Use smaller test file first
# Try with data/creditcard-tiny.csv
```

#### 5. Application Errors

**Streamlit Connection Errors:**
```bash
# Check if port 8501 is available
netstat -tulpn | grep 8501

# Try different port
streamlit run app.py --server.port 8502
```

**Cache-Related Issues:**
```bash
# Clear all caches
rm -rf cache/*
rm -rf logs/*

# Restart application
streamlit run app.py
```

### Error Recovery

The application includes automatic error recovery mechanisms:

- **Retry Logic**: Automatic retries for transient failures
- **Graceful Degradation**: Continues operation with reduced functionality
- **Error Logging**: Detailed logs in `logs/` directory for debugging
- **User Feedback**: Clear error messages with suggested solutions

### Getting Help

1. **Check Logs**: Review files in `logs/` directory for detailed error information
2. **Test with Small File**: Use `data/creditcard-tiny.csv` to isolate issues
3. **Environment Check**: Verify all environment variables are set correctly
4. **System Resources**: Ensure adequate RAM and disk space
5. **Community Support**: Check GitHub issues for similar problems

## Development and Contributing

### Development Setup

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/vstram/i2a2_extrachallenge.git
cd i2a2_extrachallenge

# Create development environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Code quality checks
black --check .
isort --check-only .
flake8 .
mypy .
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Test specific component
pytest tests/test_csv_processor.py -v

# Integration tests with small dataset
pytest tests/test_integration.py -v
```

### Architecture Design

This application is specifically designed to overcome LLM token limitations when processing large datasets:

1. **Statistical Preprocessing**: Large CSV files are converted to compact JSON summaries
2. **Agent Orchestration**: Dual-agent system processes JSON instead of raw data
3. **Streaming Interface**: Real-time feedback during long-running operations
4. **Memory Optimization**: Efficient handling of 150MB+ files
5. **Error Recovery**: Robust error handling and retry mechanisms

### Contributing Guidelines

1. **Fork and Clone**: Fork the repository and create a feature branch
2. **Development**: Follow existing code style and patterns
3. **Testing**: Add tests for new features and ensure all tests pass
4. **Documentation**: Update documentation for user-facing changes
5. **Pull Request**: Submit PR with clear description of changes

## License

MIT License - see LICENSE file for details.

## Support and Community

- **Documentation**: Complete setup guide in [INSTALL.md](INSTALL.md)
- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/vstram/i2a2_extrachallenge/issues)
- **Development**: See [CLAUDE.md](CLAUDE.md) for development guidance
- **Examples**: Test datasets included in `data/` directory

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) for the web interface
- [LangChain](https://langchain.com/) for AI agent orchestration
- [OpenAI](https://openai.com/) for cloud-based LLM capabilities
- [Ollama](https://ollama.ai/) for local LLM deployment
- [Pandas](https://pandas.pydata.org/) for data processing
- [Plotly](https://plotly.com/) for interactive visualizations

---

**Note**: This application is optimized for fraud detection analysis but can be adapted for other large-scale CSV analysis tasks. The dual-agent architecture and statistical preprocessing approach make it suitable for any scenario where raw data exceeds LLM token limitations.
