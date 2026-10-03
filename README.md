# BERT Fine-Tuning on GLUE MRPC

Fine-tuning `bert-base-uncased` for binary sentence-pair classification using the GLUE MRPC dataset and Hugging Face Transformers.

## Overview

This project demonstrates how to fine-tune a pretrained BERT model to determine whether two sentences are semantically equivalent.

The model receives two sentences:

```text
Sentence 1
Sentence 2
```

and predicts one of two classes:

* `0` — Not equivalent
* `1` — Equivalent

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Evaluate
* scikit-learn
* BERT

## Dataset

The project uses the MRPC (Microsoft Research Paraphrase Corpus) dataset from GLUE.

Dataset configuration:

```python
load_dataset("nyu-mll/glue", "mrpc")
```

## Model

```text
bert-base-uncased
```

The pretrained BERT model is fine-tuned using `AutoModelForSequenceClassification` with two output classes.

## Training

The model is trained for 3 epochs with:

```text
Learning rate: 2e-5
Training batch size: 16
Evaluation batch size: 16
Weight decay: 0.01
```

The best checkpoint is selected based on validation F1 score.

## Evaluation

The model is evaluated using:

* Accuracy
* F1 score
* Validation loss

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/bert-mrpc.git
cd bert-mrpc
```

Install the dependencies:

```bash
pip install torch transformers datasets evaluate scikit-learn
```

Run training:

```bash
python3 fine-tune.py
```

## Project Structure

```text
bert-mrpc/
├── fine-tune.py
├── README.md
└── .gitignore
```

## Purpose

This project was created as a practical exercise in Transformer fine-tuning and demonstrates the workflow of taking a pretrained language model and adapting it to a downstream NLP classification task.
