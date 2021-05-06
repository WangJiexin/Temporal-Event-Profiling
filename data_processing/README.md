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
- 1).`event_docinfor_input.pickle`: List of event_reldoc data.  
Each event_reldoc is a list, and is consist of *[event-id, event text, event occurence time, top-50 publication time, top-50 bm25 scores, top-50 doc-id, split]*. For example:  
```
In: event_docinfor_input[10028]
Out: [10028, ' Donald Trump proposes to Marla Maples and gives her a 7.5 carat diamond ring', '1991-07-02', ('1991-09-23', '1991-07-07', '1993-12-17', ...), [118.617065, 113.65213, 82.18502, ...], ('475847', '458195', '656480', ...), 'train'].
```
- 2).`doc_tempinfor_input.pickle`: A dict type data, that the key is the doc-id, and the value is the list of another dict. The *i-th* dict denotes the *'event_sent_score'* and *'normalized_temp'* information of the *i-th* temporal sentences of the document. For example:
```
In: doc_tempinfor_input['1638494']
Out: [{'event_sent_score': {'0': 0.03937406, '2944': 0.13401237},
  'normalized_temp': ['20050101', '20051231']},
 {'event_sent_score': {'0': 0.03937406, '2944': 0.13401237},
  'normalized_temp': ['20060101', '20061231']},
 {'event_sent_score': {'0': 0.4056359, '2944': 0.03333078},
  'normalized_temp': ['20100101', '20101231']}]
``` 
The first dict element of the list of doc_tempinfor_input['1638494'], denotes the information of the first temporal sentence of document '1638494', the values of 'normalized_temp' denotes the normalized temporal information of this temporal sentence, and the values of 'event_sent_score' denotes the similarity values between this temporal sentence and event '0' and  event '2944'. (Also reveals that document '1638494' is in the top-50 relevant doc-id of event '0' and  event '2944')
