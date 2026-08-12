**News Source Classification**

Integrating RAG and BiLSTM Models for News Outlet Classification

A deep learning project that classifies the source/news outlet of an article based only on its textual content. The project extends a basic Retrieval-Augmented Generation (RAG) pipeline by adding a BiLSTM-based text classification model capable of learning sequential patterns, writing style, and linguistic characteristics associated with different news sources.

The project was developed as part of a Sapienza University of Rome project by Team Insight.



**Project Overview**

The original RAG assignment focused primarily on information retrieval:

User Query
    ↓
Retrieve Related News
    ↓
Format Results
    ↓
Send to LLM

While this approach can retrieve relevant news articles, it does not inherently understand the characteristics of the news source or learn patterns in the writing style.

This project extends the original approach by introducing a supervised deep learning classifier:

News Article
    ↓
Text Cleaning
    ↓
Tokenization
    ↓
Sequence Padding
    ↓
BiLSTM
    ↓
Feature Pooling
    ↓
Dense Layer
    ↓
Softmax
    ↓
Predicted News Outlet

The extension allows the system to analyze article text and predict which news outlet it most likely originated from.

**The main objective is:**

Classify the news source from the article text alone.

**Specific objectives include:**

Clean and preprocess news article text.
Combine article titles and descriptions into a single text representation.
Identify and encode different news outlets.
Establish a classical machine learning baseline using Logistic Regression.
Develop a BiLSTM-based deep learning classifier.
Handle class imbalance using class weights.
Learn sequential and stylistic patterns in news articles.
Evaluate the model on unseen validation data.
Provide predictions with confidence scores.
Predicted News Outlet

The extension allows the system to analyze article text and predict which news outlet it most likely originated from.
