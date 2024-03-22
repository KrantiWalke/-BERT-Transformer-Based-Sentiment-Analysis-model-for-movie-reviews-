# Sentiment Analysis with Transformer Models using [![PyTorch](https://img.shields.io/badge/pytorch-%23EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Transformers](https://img.shields.io/badge/transformers-%234A329A.svg?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/transformers/) [![BERT](https://img.shields.io/badge/BERT-%234A329A.svg?style=flat-square&logo=transformers&logoColor=white)](https://huggingface.co/transformers/model_doc/bert.html) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23F7931E.svg?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/)



## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Usage](#usage)
9. [Example Usage](#example-usage)
10. [Results](#results)
11. [Acknowledgments](#acknowledgments)


## Overview
This project implements a sentiment analysis model for movie reviews using Transformer-based architectures in PyTorch. The model utilizes pre-trained Transformer models, such as BERT, for text classification tasks. The goal is to predict whether a given movie review expresses positive or negative sentiment.

## Features
- Utilizes Transformer-based architectures for sentiment analysis.
- Tokenizes and preprocesses text using the Hugging Face `transformers` library.
- Trains the sentiment analysis model on the IMDb dataset.
- Evaluates the model's performance on a held-out test set.
- Provides a function for predicting sentiment of custom text inputs.

## Requirements
[![Python](https://img.shields.io/badge/python-3.8-blue.svg?style=flat-square)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/pytorch-%23EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-%234A329A.svg?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/transformers/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-%23013243.svg?style=flat-square&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![tqdm](https://img.shields.io/badge/tqdm-%232C8EBB.svg?style=flat-square&logo=tqdm&logoColor=white)](https://github.com/tqdm/tqdm)
[![datasets](https://img.shields.io/badge/datasets-%23000.svg?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/docs/datasets/)

- Python 3.x
- PyTorch
- Transformers
- Datasets
- Matplotlib
- tqdm

Install the required dependencies with:

```bash
pip install -r requirements.txt --upgrade
```

## Dataset
The sentiment analysis model is trained on the IMDb dataset, which contains movie reviews labeled with sentiment (positive or negative). The dataset is loaded using the `datasets` library from Hugging Face.

## Model Architecture
The sentiment analysis model employs a Transformer-based architecture, specifically utilizing BERT or other Transformer models pretrained on large corpora. The final classification layer is added on top of the Transformer encoder to predict sentiment.

## Training
- The model is trained using Adam optimizer with a learning rate of 1e-5.
- Cross-entropy loss is used as the loss function.
- The training loop runs for a specified number of epochs, monitoring loss and accuracy.
- Model checkpoints are saved based on the best validation loss.

## Evaluation
- The model is evaluated on a held-out test set to assess its performance.
- Evaluation metrics include loss and accuracy.

## Usage
1. Train the model by running the provided script.
2. Evaluate the trained model using the test set.
3. Predict sentiment for custom text inputs using the provided function.

## Example Usage:
In the given script just replace/add texts and excte the script.
```python
# Example text inputs
texts = [
    "This film is terrible!",
    "This film is great!",
    "This film is not terrible, it's great!",
    "This film is not great, it's terrible!"
]
```

## Results and Conclusion
After training for multiple epochs, the model achieves a test accuracy of approximately 93%. Sample predictions demonstrate the model's ability to classify sentiment accurately.
The project aimed to build a sentiment analysis model for movie reviews using a transformer-based architecture, specifically BERT, implemented in PyTorch. Here are some key results and findings from the project:

1. **Model Performance**:
   - The trained model achieved high accuracy on both validation and test sets, with a test accuracy of approximately 93.3%. This indicates that the model is effective in predicting sentiment from movie reviews.
   - The model's performance was evaluated using both loss metrics and accuracy metrics over multiple epochs of training. It showed consistent improvement in accuracy over training epochs, indicating effective learning.

2. **Transformer Architecture**:
   - The transformer-based architecture, leveraging pre-trained BERT embeddings, demonstrated its effectiveness in capturing contextual information from movie reviews.
   - By fine-tuning the pre-trained BERT model on the sentiment analysis task, the model learned to extract relevant features and patterns from the textual data.

3. **Tokenization and Numericalization**:
   - The use of the Hugging Face `transformers` library facilitated efficient tokenization and numericalization of input text data, enabling seamless integration with the model.
   - Tokenization and numericalization were essential preprocessing steps to convert raw text into input features suitable for the transformer model.

4. **Training and Evaluation**:
   - The training process involved optimizing the model parameters using the Adam optimizer and minimizing the cross-entropy loss function.
   - The evaluation of the model was conducted on separate validation and test datasets to assess its generalization performance.
   - ```test_loss: 0.177```, ```test_acc: 0.933```

    ![image](https://github.com/KrantiWalke/-BERT-Transformer-Based-Sentiment-Analysis-model-for-movie-reviews-/assets/72568005/cfe204ce-0a17-4ecc-828b-32a33e15307e)
    ![image](https://github.com/KrantiWalke/-BERT-Transformer-Based-Sentiment-Analysis-model-for-movie-reviews-/assets/72568005/ee09112b-6719-46cf-bab4-68c171627bae)

5. **Predictions**:
   - The model was capable of making sentiment predictions on custom text inputs, demonstrating its practical utility beyond evaluation metrics.
   - Example predictions on both positive and negative sentiment phrases showcased the model's ability to distinguish between different sentiment polarities accurately.
     
   ![image](https://github.com/KrantiWalke/Transformer-Based-Sentiment-Analysis-model-for-movie-reviews/assets/72568005/bbce16d4-5dba-4601-b79a-0e96fd1cf16e)

6. **Discussion**:
   - The project highlighted the effectiveness of transformer-based models, such as BERT, in natural language processing tasks like sentiment analysis.
   - The results indicate the potential of leveraging pre-trained language models for various text classification tasks, providing robust performance out of the box with minimal fine-tuning.
   - Future work could involve further experimentation with different transformer architectures, hyperparameter tuning, and exploring additional techniques to improve model performance further.

Overall, the project demonstrated the feasibility and effectiveness of employing transformer-based models for sentiment analysis tasks, providing insights into the state-of-the-art techniques in NLP.

## Acknowledgments
- [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23F7931E.svg?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/) for providing pretrained Transformer models and datasets.
- [![PyTorch](https://img.shields.io/badge/pytorch-%23EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/) and [![Transformers](https://img.shields.io/badge/transformers-%234A329A.svg?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/transformers/) communities for their contributions to deep learning research and development.
