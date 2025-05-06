# Transformer Architectures for Political Speech Classification and Language Modeling

This project implements key components of the Transformer architecture from scratch using PyTorch. It is designed for experimenting with both encoder-based and decoder-based transformers. The models are trained on political speech data to perform classification and language modeling tasks.

## Project Structure

- `main.py`: Entry point for training and evaluation. Defines key hyperparameters and training loops.
- `dataset.py`: Dataset classes for the classification and language modeling tasks.
- `tokenizer.py`: A simple word-level tokenizer using NLTK.
- `transformer.py`: Implementation of the Transformer encoder and decoder blocks, including attention, feedforward layers, and positional embeddings.
- `utilities.py`: Utility functions for validating attention mechanisms and visualizing attention matrices.
- `speechesdataset/`: Contains all training and test datasets for classification (`CLS`) and language modeling (`LM`).

## Tasks

### 1. Transformer Encoder + Classifier
- Implemented from scratch.
- Trained jointly to classify which politician (Obama, G. W. Bush, G. H. Bush) gave a given speech segment.
- Evaluated using classification accuracy.

### 2. Transformer Decoder for Language Modeling
- GPT-style decoder trained using an autoregressive objective.
- Evaluated using perplexity on speeches from different politicians.


To run part 1, use: ```python main.py --model part1```
To run part 2, use: ```python main.py --model part2```

## Features

- Full custom implementation of attention heads, positional encodings, and feedforward layers.
- Encoder trained end-to-end for classification.
- Decoder trained on next-word prediction.
- Sanity checks and visualizations for attention matrices.
- Evaluation of encoder via accuracy and decoder via perplexity.


## Dependencies

- Python 3.8+
- PyTorch
- NLTK
- Matplotlib
