import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

def eventemb_And_doctempsentemb(event_docinfor_data):
    event_text_list=[]
    for i,event_data in enumerate(event_docinfor_data):
        event_text=event_data[1]
        event_text_list.append(event_text)
    event_sentence_embeddings = model.encode(event_text_list)
    
    doctempsentidx_sent_dict = pickle.load(open("../data/filtered_doctempsentidx_sent_dict.pickle", 'rb'))
    print(len(doctempsentidx_sent_dict))
    doc_sent_list = []
    doctempsentidx_list = []
    for doctempsentidx,sent in doctempsentidx_sent_dict.items():
        doctempsentidx_list.append(doctempsentidx)
        doc_sent_list.append(sent)
    sentence_embeddings = model.encode(doc_sent_list)
    doctempsentidx_emb_dict = dict()
    for doctempsentidx,sent_emb in zip(doctempsentidx_list,sentence_embeddings):
        doctempsentidx_emb_dict[doctempsentidx] = sent_emb
    return event_sentence_embeddings,doctempsentidx_emb_dict

def sim_calculation(event_docinfor_data,event_sentence_embeddings,doctempsentidx_emb_dict):
    for event_data in event_docinfor_data:
        event_idx=event_data[0]
        event_text=event_data[1]
        event_text_embeddings = event_sentence_embeddings[i].reshape(1, -1)
        event_docinfo_list=event_data[-1]
        doc_sent_embeddings_list=[]
        for event_docinfo in event_docinfo_list:
            doc_idx = event_docinfo[0]
            if doc_idx not in doc_temp_dict:
                continue
            doc_temp_list=doc_temp_dict[doc_idx]
            for doc_temp in doc_temp_list:
                sent_idx=doc_temp['tempsent_idx']
                doctempsentidx=doc_idx+"_"+str(sent_idx)
                sent_emb = doctempsentidx_emb_dict[doctempsentidx]
                doc_sent_embeddings_list.append(sent_emb)
        doc_sent_embeddings_array=np.array(doc_sent_embeddings_list)
        similarity_result_matrix=cosine_similarity(event_text_embeddings,doc_sent_embeddings_array)[0]
        similarity_idx=0
        for event_docinfo in event_docinfo_list:
            doc_idx = event_docinfo[0]
            if doc_idx not in doc_temp_dict:
                continue
            doc_temp_list=doc_temp_dict[doc_idx]
            for doc_temp in doc_temp_list:
                sent_idx=doc_temp['tempsent_idx']
                doctempsentidx=doc_idx+"_"+str(sent_idx)
                doc_temp["event_sent_score"][str(event_idx)]=similarity_result_matrix[similarity_idx]
                similarity_idx+=1
    doctemp_with_sim_dict = doc_temp_dict
    return doc_temp_dict

def save_event2tempsent_siminfor():
    event_docinfor_data = pickle.load(open("../data/event_data_with_docinfor.pickle", 'rb'))
    for record_i,record in enumerate(event_docinfor_data):
        record.insert(0,record_i)
    event_sentence_embeddings,doctempsentidx_emb_dict = eventemb_And_doctempsentemb(event_docinfor_data)
    
    doc_temp_dict = pickle.load(open("../data/filtered_doc_temp_dict.pickle", 'rb'))
    for event_record in event_docinfor_data:
        event_idx = event_record[0]
        for r in event_record[-1]:
            docid = r[0]
            if docid not in doc_temp_dict:
                continue
            temp_dict_list = doc_temp_dict[docid]
            for dict_i,temp_dict in enumerate(temp_dict_list):
                if 'event_sent_score' not in temp_dict:
                    temp_dict['event_sent_score'] = dict()
                temp_dict['event_sent_score'][str(event_idx)]=0.0
    doc_temp_dict = sim_calculation(event_docinfor_data,event_sentence_embeddings,doctempsentidx_emb_dict)
    pickle.dump(doc_temp_dict, open("../data/filtered_doctemp_with_sim_dict.pickle", "wb"))
    
if __name__ == "__main__":
    save_event2tempsent_siminfor()
