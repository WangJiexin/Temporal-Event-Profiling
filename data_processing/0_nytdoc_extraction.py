import os
import re
import pke
import pickle
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
new_stop_words_set={"year","years","day","days","event","occur",
                    "occurred","name","names","named","happen",
                    "happened","caused","known","country","city","times","accident"}
STOP_WORDS.update(new_stop_words_set)

def yake_keyword_extraction(sentence_text,stoplist,ngram=2,top_n_keywords=15):
    extractor = pke.unsupervised.YAKE()
    extractor.load_document(input=sentence_text,language='en',normalization=None)
    extractor.candidate_selection(n=ngram, stoplist=stoplist)
    window = 2
    use_stems = False
    extractor.candidate_weighting(window=window,stoplist=STOP_WORDS,use_stems=use_stems)
    threshold = 0.8
    keyphrases = extractor.get_n_best(n=top_n_keywords, threshold=threshold)
    keywords_list=[]
    for keyword_val_tuple in keyphrases:
        keywords_list.append(keyword_val_tuple[0])
    return keywords_list

def ir_method(keywords_list):
    #Please define your own IR method, and use NYT corpus as the knowledge source.
    #To obtain the corpus, refer to https://catalog.ldc.upenn.edu.
    #In the paper, we use ElasticSearch as the default IR method to extract top-50 articles.
    keywords_string=""
    for k in keywords_list:
        keywords_string=keywords_string+r"\""+k+r"\"|"
    keywords_string=keywords_string[:-2]+"\""
    elastic_curl_command=r"""curl -u XXXnytcorpus_commandXXX?scroll=10m&size=50' ¥ -H 'Content-Type: application/json' ¥ --data-raw '{"query": {"simple_query_string":{"query":"query_keywords_string","fields" : ["body"]}}}'"""
    #please define your own elastic_curl_command (replace XXXnytcorpus_commandXXX)
    elastic_command=elastic_curl_command.replace("query_keywords_string",keywords_string)
    
    es_result = os.popen(elastic_command).readlines()
    doc_info_list=[]
    body_text_list=[]
    for doc_text in es_result[0].split(r'_type":"_doc')[1:]:
        doc_info=re.findall(r'_id":"(\d+)","_score":(\d+\.\d+).*"publicationDate":"(\d+-\d+-\d+)"',doc_text)
        #doc_info: (docid, bm25_score,doc_pubdate), for example: ('1224070', '13.3322735', '2000-08-20')
        body_text = re.findall(r'"body":"(.*)","articleAbstract"',doc_text)
        #body_text: main article content
        doc_info_list.extend(list(doc_info))
        body_text_list.append(body_text[0].replace("\\n","\n"))
    return doc_info_list,body_text_list

def nytdoc_extraction():
    event_data = pd.read_pickle("../data/dataset/event_data.pickle").values.tolist()
    docid_doctext_dict = dict()
    for i,record in enumerate(event_data):
        question=record[0]
        keywords_list=yake_keyword_extraction(question,STOP_WORDS)
        doc_info_list,body_text_list=ir_method(keywords_list)
        record.append(doc_info_list)
        for doc_i, doc_infor in enumerate(doc_info_list):
            docid_doctext_dict[doc_infor[0]] = body_text_list[doc_i]
    #save two kinds of important data
    pickle.dump(event_data, open("../data/event_data_with_docinfor.pickle", "wb"))
    pickle.dump(docid_doctext_dict, open("../data/docid_doctext_dict.pickle", "wb"))

if __name__ == "__main__":
    nytdoc_extraction()
