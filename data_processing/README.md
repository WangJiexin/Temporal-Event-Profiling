# Data Processing

This folder contains the code for preparing multivariate time series and necessary data input for TEP-Trans model, which is consist of four steps.

## 1. Document Extraction (0_nytdoc_extraction.py)
The first step is to identify keywords for each event description and use them to retrieve top k relevant news articles from the NYT corpus. 
#Please define your own IR method, and use NYT corpus as the knowledge source.
    #To obtain the corpus, refer to https://catalog.ldc.upenn.edu.
    #In the paper, we use ElasticSearch as the default IR method to extract top-50 articles.

## 2. Document Content Temporal Information Extraction & Normalization

## 3. Sentence2event Similarity Calculation

## 4. Model Input Data Preparation

