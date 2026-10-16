# 🗜️ Edge-Optimized Neural Compression Engine

A PyTorch-based Convolutional Autoencoder pipeline designed to compress image data into dense latent representations, featuring Post-Training Quantization (PTQ) for efficient inference on edge devices.

This project demonstrates the end-to-end lifecycle of an Applied AI system: defining a custom CNN architecture, training on hardware-accelerated backends (Apple Metal/MPS), and compiling the model from 32-bit floating-point to 8-bit integers to drastically reduce memory footprint without losing reconstruction fidelity.

## 📊 Optimization Metrics

By utilizing PyTorch's FX Graph Quantization (QNNPACK backend), the model was successfully optimized for edge-device RAM constraints:

| Metric | Unoptimized (FP32) | Edge-Optimized (INT8) | Improvement |
|---|---|---|---|
| **Memory Footprint** | 188.99 KB | 55.24 KB | **70.77% Reduction** |
| **Data Bottleneck** | 12,288 elements | 4,096 elements | **3.00X Compression** |

## 🏗️ System Architecture

1. **Convolutional Encoder:** Progressively downsamples (strided convolutions) input images into a dense latent tensor bottleneck.
2. **FX Quantization Compiler:** Programmatically traces the PyTorch computational graph, calibrates activation ranges, and swaps heavy FP32 operations for lightweight INT8 instructions.
3. **Inference Pipeline:** Executes the 8-bit model to extract the latent bottleneck, saves it to disk as a binary `.bin` file, and reconstructs the image via the transposed convolutional decoder.

## 🚀 Quick Start Guide

### 1. Installation
```bash
git clone [https://github.com/YourUsername/edge-neural-compressor.git](https://github.com/YourUsername/edge-neural-compressor.git)
cd edge-neural-compressor
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision numpy Pillow

2. Train the Autoencoder
Train the FP32 model locally (utilizes MPS on Apple Silicon or CUDA if available):
python train.py

3. Compile to Edge (INT8)
Quantize the trained model weights to reduce the memory footprint by ~70%:
python quantize.py

4. Run End-to-End Compression
Compress a sample image to a binary latent space and reconstruct it:
python inference.py

