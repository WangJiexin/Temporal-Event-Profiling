# Data Processing

This folder contains the code for preparing multivariate time series and necessary data input for TEP-Trans model, which is consist of four steps.

## 1. Document Extraction (1_nytdoc_extraction.py)
The first step is to identify keywords for each event description and use them to retrieve top k relevant news articles from the NYT corpus. In the paper, we use ElasticSearch as the default IR method to extract top 50 articles.
Note that due to the NYT corpus policy, we hide the ElasticSearch command in `ir_method(keywords_list)` to prevent exposing the NYT corpus to the public. To obtain the corpus, refer to https://catalog.ldc.upenn.edu. Then, please define your own IR method (modify `ir_method(keywords_list)`) that use the NYT corpus as the knowledge source.

## 2. Document Content Temporal Information Extraction & Normalization (2_tempinfor_extraction_And_normalization.py)
The second step is to extract and normalize the temporal information from the content of each retrieved document. In addition, the sentences that contain temporal information are also stored that will be used in the next step.

## 3. Sentence2event Similarity Calculation (3_event2tempsent_simcalculation.py)
The third step is to calculate the similarity between event description and the temporal sentences of its relevant sentences. 

## 4. Model Input Data Preparation (4_preprare_input_data.py)
The final step prepares two kinds of data, that can be easily transformed to multivariate time series, which are the input of TEP-Trans model. The two kinds of data are:
- `doc_tempinfor_input.pickle`: The event description.
[0, ' NFC Championship, Trans World Dome, St. Louis: St. Louis Rams beat Tampa Bay Buccaneers, 11-6', '2000-01-23', ('2000-01-24','2000-01-24','2000-01-24',...),
[68.32553, 65.53658,59.50751,...],  ('1171168','1171143','1171144',...),'train']

- `event_docinfor_input.pickle`: The event description.
