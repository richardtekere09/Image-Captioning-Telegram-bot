# Image Captioning Telegram Bot

A comprehensive image captioning system using deep learning, combining a Telegram bot (aiogram) with a FastAPI inference service. This project uses the BLIP (Bootstrapped Language Image Pretraining) model for generating natural language descriptions of images.

## 🎯 Project Overview

This coursework project demonstrates:
- **Deep Learning**: State-of-the-art image captioning using BLIP model
- **Microservices Architecture**: Separation of bot logic and ML inference
- **Modern Python**: aiogram 3.x, FastAPI, and async programming
- **Performance Analysis**: Comprehensive benchmarking of inference speed and memory usage

## 🏗️ Architecture

```
User → Telegram Bot (aiogram) → FastAPI Service → BLIP Model → Caption → User
```

### Components:

1. **Telegram Bot** (`telegram_bot.py`): 
   - User interface via Telegram
   - Handles commands and photo uploads
   - Communicates with FastAPI service

2. **FastAPI Service** (`fastapi_service.py`):
   - REST API for image captioning
   - Loads and manages BLIP model
   - Provides single and batch inference endpoints

3. **Benchmark Tool** (`benchmark.py`):
   - Performance evaluation across different batch sizes
   - Generates metrics and visualizations
   - Produces detailed reports

## 📋 Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- 8GB+ RAM (16GB recommended for large model)
- GPU recommended for faster inference

## 📊 Benchmarking

Run the benchmark to evaluate model performance:

```bash
python benchmark.py
```

This will:
- Test inference with batch sizes: 1, 2, 4, 8, 16
- Run 5 iterations per batch size
- Measure inference time and memory usage
- Generate visualization plots
- Save results to JSON file

### Benchmark Output

- **Console**: Formatted table with results
- **Plot**: `benchmark_results.png` - Visual comparison
- **JSON**: `benchmark_results.json` - Detailed metrics


## 📁 Project Structure

```
image-captioning-bot/
├── telegram_bot.py          # Telegram bot implementation
├── fastapi_service.py       # FastAPI inference service
├── benchmark.py             # Performance benchmarking tool
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── .env                    # Your configuration (create this)
├── README.md               # This file
└── run.sh / run.bat        # Startup scripts
```

## 🎓 Academic Information

### Model: BLIP (Bootstrapped Language Image Pretraining)

- **Paper**: Li et al. (2022) - "BLIP: Bootstrapping Language-Image Pre-training"
- **Architecture**: Vision Transformer + Text Decoder
- **Training**: MS COCO and other large-scale datasets
- **Capabilities**: Image captioning, VQA, image-text matching

### Evaluation Metrics

Common metrics for image captioning:
- **BLEU**: Precision-based metric comparing n-grams
- **METEOR**: Considers synonyms and paraphrasing
- **CIDEr**: Consensus-based metric for image description
- **SPICE**: Semantic propositional content metric

### Loss Functions

- **Cross-Entropy Loss**: Standard for sequence generation
- **Label Smoothing**: Prevents overconfidence
- **CIDEr Optimization**: Direct optimization of evaluation metric

##  Report Guidelines
 report, include:

1. **Introduction**: Overview of image captioning and project goals
2. **Literature Review**: BLIP, Show and Tell, attention mechanisms
3. **Methodology**: Architecture, implementation details
4. **Experiments**: Benchmark results, performance analysis
5. **Results**: Tables and graphs from benchmark
6. **Discussion**: Findings, limitations, future work
7. **Conclusion**: Summary of achievements
8. **References**: Academic papers and documentation

##  Future Enhancements

- Add support for multiple models (GIT, OFA, etc.)
- Implement user preferences and history
- Add multilingual caption support
- Fine-tune model on custom datasets
- Deploy to cloud (AWS, Azure, Google Cloud)
- Add web interface alongside Telegram
- Implement caching for frequently requested images

##  References

1. Li, J., et al. (2022). BLIP: Bootstrapped Language-Image Pretraining. arXiv:2201.12086
2. Vinyals, O., et al. (2015). Show and Tell: A Neural Image Caption Generator. CVPR
3. Xu, K., et al. (2016). Show, Attend and Tell. ICML
4. Chen, X., et al. (2015). Microsoft COCO Captions. arXiv:1504.00325

