import pickle

def prepare_input_data():
	doc_temp_dict = pickle.load(open("../data/filtered_doctemp_with_sim_dict.pickle", 'rb'))
	doc_normalizedtemp_dict = pickle.load(open("../data/doc_normalizedtemp_dict.pickle", 'rb'))
	doc_tempinfor_input = dict()
	for docid,tempdict_list in doc_temp_dict.items():
	    normalizedtemp_list = doc_normalizedtemp_dict[docid]
	    assert len(normalizedtemp_list)==len(tempdict_list)
	    doc_tempinfor_list = []
	    for temp_i,tempdict in enumerate(tempdict_list):
	        tempsent_dict = dict()
	        normalized_temp = normalizedtemp_list[temp_i]
	        tempsent_dict['event_sent_score'] = tempdict['event_sent_score']
	        tempsent_dict['normalized_temp'] = normalized_temp
	        doc_tempinfor_list.append(tempsent_dict)
	    doc_tempinfor_input[docid] = doc_tempinfor_list

	event_docinfor_data = pickle.load(open("../data/event_data_with_docinfor.pickle", 'rb'))
	for record_i,record in enumerate(event_docinfor_data):
	    record.insert(0,record_i)
	event_docinfor_input = []
	for record in event_docinfor_data:
	    record_idx, event_text, event_time, split, docinfor_list = record
	    docinfor_lists = list(zip(*docinfor_list))
	    docid_list,bm25_list,doc_pubdate = docinfor_lists
	    bm25_list = list(map(float, bm25_list))
	    input_list = [record_idx,event_text,event_time,doc_pubdate,bm25_list,docid_list,split]
	    event_docinfor_input.append(input_list)

	pickle.dump(doc_tempinfor_input, open("../data/input_data/doc_tempinfor_input.pickle", "wb"))
	pickle.dump(event_docinfor_input, open("../data/input_data/event_docinfor_input.pickle", "wb"))

if __name__ == "__main__":
    prepare_input_data()
