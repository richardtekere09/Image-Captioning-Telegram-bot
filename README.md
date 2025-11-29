# 🤖 Image Captioning Telegram Bot

A production-ready image captioning system using deep learning, combining a Telegram bot (aiogram) with a FastAPI inference service powered by the BLIP model.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![License](https://img.shields.io/badge/license-MIT-blue)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Docker](#docker)
- [Benchmarking](#benchmarking)
- [Project Structure](#project-structure)

## ✨ Features

- 🤖 **Telegram Bot Interface**: User-friendly interaction via Telegram
- 🧠 **BLIP Model**: State-of-the-art image captioning using pretrained model
- 🚀 **FastAPI Service**: High-performance REST API for inference
- 🛡️ **Rate Limiting**: Anti-spam protection with configurable throttling
- 📊 **Comprehensive Benchmarking**: Performance analysis across batch sizes
- 🐳 **Docker Support**: Containerized deployment with docker-compose
- ✅ **Production-Ready**: Error handling, logging, monitoring
- 📝 **Well-Documented**: Extensive documentation and type hints

## 🏗️ Architecture

```
┌─────────────────┐
│  Telegram User  │
└────────┬────────┘
         │ Sends Image
         ↓
┌─────────────────────────────────────┐
│    Telegram Bot (aiogram)          │
│  ┌──────────────────────────────┐  │
│  │ Middleware Layer             │  │
│  │  • Rate Limiting             │  │
│  │  • Logging                   │  │
│  │  • Error Handling            │  │
│  └──────────────────────────────┘  │
└─────────┬───────────────────────────┘
          │ HTTP POST
          ↓
┌─────────────────────────────────────┐
│   FastAPI Service                   │
│  ┌──────────────────────────────┐  │
│  │ ML Module                    │  │
│  │  • Model Loader              │  │
│  │  • Preprocessing             │  │
│  │  • Inference Engine          │  │
│  └──────────────────────────────┘  │
└─────────┬───────────────────────────┘
          │ Returns Caption
          ↓
      User receives caption
```

## 📦 Prerequisites

### Hardware Requirements

**Minimum** (CPU-only):
- CPU: Intel i5 or AMD Ryzen 5
- RAM: 8 GB
- Storage: 5 GB free space

**Recommended** (with GPU):
- GPU: NVIDIA GTX 1650+ (4GB VRAM)
- RAM: 16 GB
- Storage: 10 GB free space

### Software Requirements

- Python 3.11+
- pip 23.0+
- Git
- (Optional) Docker & Docker Compose
- (Optional) CUDA 11.8+ for GPU support

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/image-captioning-bot.git
cd image-captioning-bot
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 4. Get Telegram Bot Token

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Follow instructions to create your bot
4. Copy the bot token (looks like: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)

### 5. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env file and add your bot token
nano .env  # or use any text editor
```

### 6. Test Configuration

```bash
# Test that configuration loads correctly
python config.py
```

Expected output:
```
====================================================
CONFIGURATION TEST
====================================================
✅ Configuration loaded successfully!
```

## ⚙️ Configuration

Edit `.env` file:

```env
# Required
BOT_TOKEN=your_bot_token_here

# Optional (defaults shown)
RATE_LIMIT_SECONDS=2.0
MAX_IMAGE_SIZE=10485760
API_URL=http://localhost:8000
MODEL_NAME=Salesforce/blip-image-captioning-base
DEVICE=auto
LOG_LEVEL=INFO
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `BOT_TOKEN` | *Required* | Telegram bot token from @BotFather |
| `RATE_LIMIT_SECONDS` | `2.0` | Seconds between requests per user |
| `MAX_IMAGE_SIZE` | `10485760` | Max image size in bytes (10MB) |
| `API_URL` | `http://localhost:8000` | FastAPI service URL |
| `MODEL_NAME` | `Salesforce/blip-image-captioning-base` | Hugging Face model |
| `DEVICE` | `auto` | Device: auto, cuda, or cpu |
| `LOG_LEVEL` | `INFO` | Logging level |

## 🎮 Usage

### Running Locally

#### Terminal 1: Start API Service

```bash
python fastapi_service.py
```

Expected output:
```
INFO:     Loading model...
INFO:     Model loaded successfully!
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Terminal 2: Start Telegram Bot

```bash
python telegram_bot.py
```

Expected output:
```
INFO:     Bot started successfully!
INFO:     Listening for messages...
```

#### Test in Telegram

1. Open Telegram
2. Search for your bot (username you created with @BotFather)
3. Send `/start` command
4. Send any image
5. Receive caption!

### Using the Bot

**Commands:**
- `/start` - Welcome message and instructions
- `/help` - Show available commands
- Send any image - Get caption

**Example Interaction:**
```
You: [sends photo of a cat]
Bot: 🔄 Processing your image...
Bot: 📝 A cat sitting on a couch.
     ⏱️ Processed in 1.23s
```

## 🛠️ Development

### Project Setup (for contributors)

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run code formatting
black .

# Run linting
flake8 .

# Run type checking
mypy telegram_bot.py fastapi_service.py
```

### Making Changes

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes...
# ... code ...

# Run tests
pytest tests/

# Format code
black .

# Commit with conventional commit message
git commit -m "feat: add new feature description"

# Push to remote
git push origin feature/your-feature-name
```

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_ml.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

### Manual Testing

```bash
# Test config
python config.py

# Test ML model loading
python -c "from ml.model_loader import get_model; model = get_model(); print('✅ Model works!')"

# Test API endpoint
curl http://localhost:8000/health
```

## 🐳 Docker

### Build and Run with Docker Compose

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Individual Containers

```bash
# Build API container
docker build -f docker/api.Dockerfile -t caption-api .

# Build Bot container
docker build -f docker/bot.Dockerfile -t caption-bot .

# Run API
docker run -p 8000:8000 --env-file .env caption-api

# Run Bot
docker run --env-file .env caption-bot
```

## 📊 Benchmarking

### Run Benchmarks

```bash
python benchmark.py
```

This will:
- Test batch sizes: 1, 2, 4, 8, 16
- Measure inference time
- Measure memory usage
- Generate visualization
- Save results to `results/benchmark_results.json`

### View Results

```bash
# View JSON results
cat results/benchmark_results.json

# View plot
open results/benchmark_results.png
```

## 📁 Project Structure

```
image-captioning-bot/
├── config.py                 # Configuration management
├── telegram_bot.py           # Telegram bot implementation
├── fastapi_service.py        # FastAPI inference service
├── benchmark.py              # Performance benchmarking
│
├── ml/                       # Machine Learning module
│   ├── model_loader.py       # Model initialization
│   ├── inference.py          # Caption generation
│   └── preprocessing.py      # Image preprocessing
│
├── middleware/               # Bot middleware
│   ├── throttling.py         # Rate limiting
│   ├── logging.py            # Request logging
│   └── error_handler.py      # Error handling
│
├── utils/                    # Utility functions
│   ├── image_utils.py        # Image processing
│   ├── api_client.py         # HTTP client
│   └── exceptions.py         # Custom exceptions
│
├── docker/                   # Docker configuration
│   ├── bot.Dockerfile        # Bot container
│   ├── api.Dockerfile        # API container
│   └── docker-compose.yml    # Multi-container setup
│
├── tests/                    # Test suite
│   ├── test_ml.py
│   ├── test_api.py
│   └── test_bot.py
│
└── requirements.txt          # Python dependencies
```

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [BLIP](https://github.com/salesforce/BLIP) - Salesforce Research
- [aiogram](https://github.com/aiogram/aiogram) - Telegram Bot framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Transformers](https://huggingface.co/transformers/) - Hugging Face

## 📧 Contact

For questions or issues, please open a GitHub issue or contact: your.email@example.com

---

**Built with ❤️ for coursework project**
