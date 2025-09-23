# Installation Guide

This guide provides detailed instructions for installing the AI Analytics Chatbot with different dependency configurations.

## Prerequisites

- Python 3.11 or higher
- pip package manager
- Git (for development installation)

## Quick Start

### Basic Installation

For basic functionality with OpenAI support:

```bash
pip install -r requirements.txt
```

### Installation with Optional Features

#### Full Installation (All Features)
```bash
pip install -e .[all]
```

#### Specific Feature Sets

**Ollama Support (Local LLM)**
```bash
pip install -e .[ollama]
```

**Performance Monitoring**
```bash
pip install -e .[performance]
```

**Advanced Features**
```bash
pip install -e .[advanced]
```

**Development Tools**
```bash
pip install -e .[dev]
```

**Production Deployment**
```bash
pip install -e .[production]
```

## Detailed Installation Options

### 1. Requirements Files Method

#### Core Requirements Only
```bash
pip install -r requirements.txt
```

#### Optional Enhancements
```bash
pip install -r requirements-optional.txt
```

#### Development Environment
```bash
pip install -r requirements-dev.txt
```

#### Production Deployment
```bash
pip install -r requirements-production.txt
```

### 2. Modern Python Package Installation

#### Install from PyPI (when published)
```bash
pip install i2a2-extrachallenge
```

#### Install with Specific Features
```bash
# Local LLM support
pip install i2a2-extrachallenge[ollama]

# Performance monitoring
pip install i2a2-extrachallenge[performance]

# All features
pip install i2a2-extrachallenge[all]
```

### 3. Development Installation

#### Clone and Install
```bash
git clone https://github.com/vstram/i2a2_extrachallenge.git
cd i2a2_extrachallenge
pip install -e .[dev]
```

#### Pre-commit Hooks (Recommended for Contributors)
```bash
pre-commit install
```

## Environment Setup

### 1. Virtual Environment (Recommended)

#### Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Using conda
```bash
conda create -n ai-analytics python=3.11
conda activate ai-analytics
pip install -r requirements.txt
```

### 2. Environment Variables

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# LLM Configuration
OPENAI_API_KEY=your-openai-api-key-here
LLM_PROVIDER=openai

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
CACHE_ENABLED=true
```

## Platform-Specific Instructions

### Windows

#### Installation
```cmd
# Using Command Prompt
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Using PowerShell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Potential Issues
- **Long Path Names**: Enable long path support in Windows
- **Visual C++ Build Tools**: Install Microsoft C++ Build Tools if compilation errors occur

### macOS

#### Installation
```bash
# Using Homebrew (if Python not installed)
brew install python@3.11

# Standard installation
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### M1/M2 Mac Considerations
- Use `python3.11` explicitly
- Some packages may require ARM64 builds

### Linux

#### Ubuntu/Debian
```bash
# Install Python and dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Create environment and install
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### CentOS/RHEL/Fedora
```bash
# Install Python
sudo dnf install python3.11 python3.11-venv python3.11-devel

# Create environment and install
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Docker Installation

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d --build
```

### Manual Docker Build

```bash
# Build image
docker build -t ai-analytics-chatbot .

# Run container
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key ai-analytics-chatbot
```

## Verification

### Test Installation

```bash
# Test basic functionality
python -c "import streamlit, pandas, langchain; print('✅ Core packages installed')"

# Test optional packages
python -c "
try:
    import psutil, sklearn, matplotlib
    print('✅ Optional packages installed')
except ImportError as e:
    print(f'⚠️  Some optional packages missing: {e}')
"

# Run application test
streamlit run app.py --server.headless true --server.port 8501
```

### Verify Environment

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Check configuration
python -c "from config.settings import get_settings; print(get_settings())"
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Reinstall packages
pip install --force-reinstall -r requirements.txt

# Clear pip cache
pip cache purge
```

#### Permission Errors
```bash
# Use user installation
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
```

#### Version Conflicts
```bash
# Create fresh environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Memory Issues
```bash
# Install with no cache
pip install --no-cache-dir -r requirements.txt

# Or increase pip memory
pip install --progress-bar off -r requirements.txt
```

### System Requirements

#### Minimum Requirements
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **CPU**: 2 cores (4 cores recommended)

#### Recommended Requirements
- **RAM**: 16GB for large datasets
- **Storage**: 10GB for caching and logs
- **CPU**: 8 cores for optimal performance

### Performance Optimization

#### For Large Datasets
```bash
# Install performance packages
pip install -r requirements-optional.txt

# Enable performance monitoring
export ENABLE_MONITORING=true
export CACHE_MAX_SIZE_MB=1000
```

#### For Production
```bash
# Install production requirements
pip install -r requirements-production.txt

# Set production environment
export ENVIRONMENT=production
export LOG_LEVEL=INFO
```

## Next Steps

After successful installation:

1. **Configure Environment**: Set up your `.env` file
2. **Test LLM Connection**: Verify OpenAI or Ollama connectivity
3. **Upload Test Data**: Try with a small CSV file
4. **Explore Features**: Check out the README for usage instructions

## Support

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review the [GitHub issues](https://github.com/vstram/i2a2_extrachallenge/issues)
3. Create a new issue with your environment details

## Contributing

For development setup and contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).