# CS117-AI-extract-unconstructed-orders

This project focuses on extracting structured information from unstructured agricultural order messages using AI techniques. The goal is to convert raw text messages about agricultural orders into structured JSON format that can be easily processed by downstream systems.

## Project Overview

The system processes agricultural order messages written in Vietnamese and extracts key information such as:
- Customer name
- Phone number
- Location tag
- Payment intent
- Delivery notes
- Order notes
- Items (product, quantity, budget)
- Inquiries

## Model

The project utilizes a fine-tuned language model based on Qwen3-4B-Instruct, with LoRA (Low-Rank Adaptation) applied for efficient fine-tuning on the agricultural order extraction task.

**Hugging Face Model:** [zxcvmh666/model_gAO_extract](https://huggingface.co/zxcvmh666/model_gAO_extract)

## Files in this Repository

- `preprocess-data-tn.py`: Script for preprocessing training data
- `preprocess-test-data.py`: Script for preprocessing test data
- `qwen3-4in-315-final.ipynb`: Jupyter notebook containing the full fine-tuning and evaluation pipeline
- `qwen3-instruct.ipynb`: Jupyter notebook for instruction fine-tuning
- `data/`: Directory containing training and test datasets in various formats (CSV, JSONL)
- `test_evaluation_log-*.jsonl`: Evaluation logs showing model performance

## Usage

1. Clone this repository
2. Install required dependencies (see notebooks for specific versions)
3. Run the preprocessing scripts to prepare your data
4. Use the fine-tuned model from Hugging Face for inference, or fine-tune your own using the provided notebooks

## Results

The model achieves high accuracy in extracting structured information from unstructured agricultural order messages, as demonstrated in the evaluation logs included in this repository.
