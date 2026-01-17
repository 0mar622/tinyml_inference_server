# TinyML Inference Server

A C++ backend application that implements an HTTP-based image inference server using ONNX Runtime. The server accepts raw image bytes over HTTP, preprocesses the image, executes a trained convolutional neural network on CPU, and returns the predicted class, confidence score, and inference latency. The model was trained in PyTorch and exported to ONNX for efficient C++ inference. Built to strengthen systems-level programming, model deployment, and machine learning inference infrastructure fundamentals.

## Features

- C++ HTTP server using a lightweight header-only library
- Loads ONNX model once at startup for efficient inference
- Accepts raw image bytes via HTTP POST requests
- Image preprocessing pipeline implemented with OpenCV
- CNN inference using ONNX Runtime (CPU execution)
- Softmax probability computation for confidence scoring
- Returns JSON responses containing label, confidence, and latency
- Single-threaded execution for predictable performance
- Clear separation between server logic and inference logic

## Tech Stack

C++ | Linux | ONNX Runtime | OpenCV | HTTP | CNN Inference | Model Deployment

## Status

**Completed – Core functionality implemented.**

The server reliably handles HTTP requests, decodes and preprocesses image data, executes CNN inference using ONNX Runtime, and returns structured prediction results with measured latency. The project is stable for local testing and demonstrates a complete end-to-end inference pipeline from client request to model prediction.

## What I Learned

This project helped me understand how trained machine learning models are deployed and executed in systems-level C++ environments. I learned how inference runtimes load and execute models, how tensors and memory are managed explicitly, how preprocessing pipelines must exactly match training-time behavior, and how trained models can be exported from PyTorch and integrated into production-style C++ servers. The project also reinforced concepts around HTTP-based services, runtime performance measurement, and building efficient inference backends.
