# DisViTDet: A Multi-Modal Vision Transformer for Disinformation Detection

DisViTDet is a state-of-the-art deep learning model designed to classify social media posts as real or fake news. It leverages a Vision Transformer (ViT) architecture integrated with multi-modal data inputs—text, meme images, and emojis—to capture intricate relationships between these diverse content types.

## Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Data Preparation](#data-preparation)
* [Training](#training)
* [Testing](#testing)
* [Demo](#demo)
* [Model Architecture](#model-architecture)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)

## Introduction

With the proliferation of social media, disinformation in the form of fake news has become a significant challenge, impacting political stability, public health, and societal harmony. DisViTDet addresses this issue by introducing a novel multi-modal detection framework that combines textual, visual, and emoji data inputs.

## Features

* **Cross-Attention Mechanism**: Aligns meme images, textual content, and emoji embeddings within a unified architecture.
* **Transformer-Based Emoji Embeddings**: Captures sentiment cues often overlooked in fake news detection.
* **Feature Similarity Thresholding**: Quantifies inconsistencies across modalities to serve as a robust indicator of disinformation.
* **Vision Transformer (ViT) Backbone**: Provides powerful image feature extraction, surpassing traditional CNN models.

## Installation

To set up the DisViTDet project locally, follow these steps:

bash
git clone (link unavailable)
cd DisViTDet
python -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate
pip install -r requirements.txt


*Usage*

*Data Preparation*

Prepare your dataset containing meme images, text content, and emojis. Structure your data in a format compatible with the model (see `src/datasets/social_media_dataset.py` for details).

Run the data preparation script:


bash
python scripts/prepare_data.py


*Training*

To train the model:


bash
bash scripts/train.sh


This script runs the training pipeline based on the configuration specified in `experiments/config_disvitdet.json`.

*Testing*

To test the model:


bash
bash scripts/test.sh


The results, including metrics and visualizations, will be logged.

*Demo*

To run a demo with dummy data:


bash
python src/demo.py


*Model Architecture*

The DisViTDet model architecture comprises the following key components:

- *Meme Embedding*: Extracts features from meme images using a Vision Transformer (ViT).
- *Text Embedding*: Embeds textual content using a pre-trained BERT model.
- *Emoji Embedding*: Encodes emojis into a dense vector space using transformer-based embeddings.
- *Cross-Attention*: Aligns and combines features across different modalities.
- *Transformer Encoder*: Processes the combined features for classification.
- *Classifier*: Outputs the final prediction (real or fake news).

For detailed implementation, refer to the `src/models/disvitdet.py` file.

*Results*

Extensive experiments demonstrate that DisViTDet significantly outperforms existing methods on diverse datasets, validating its efficacy in real-world scenarios. The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

*Contributing*

Contributions are welcome! If you would like to contribute to DisViTDet, please fork the repository and submit a pull request with your improvements.

*License*

This project is licensed under the Author's License - see the LICENSE file for details.