import re
import pickle
import pandas as pd
import spacy
from sutime import SUTime
nlp = spacy.load("en_core_web_sm")
sutime = SUTime(mark_time_ranges=True, include_range=True)

def basictempinfor_fun(event_docinfor_data,docid_doctext_dict):
    doc_temp_dict = dict()
    doctempsentidx_sent_dict = dict()
    for idx,record in enumerate(event_docinfor_data):
        docinfor_list = record[-1]
        for docinfor in docinfor_list:
            docid,bm25,docpub = docinfor
            body_text = docid_doctext_dict[docid]
            sutime_dict_candidate_list=sutime.parse(body_text,docpub)
            doc_temp_dict[docid]=sutime_dict_candidate_list
            text_spacy=nlp(body_text)
            sent_list=list(text_spacy.sents)
            sent_pos_list=[]
            for sent in sent_list:
                sent_pos_list.append((sent[0].idx,sent[-1].idx+len(sent[-1].text)))
            for tempsent_idx,temp_info in enumerate(sutime_dict_candidate_list):
                start_idx=temp_info['start']
                end_idx=temp_info['end']
                temp_info['tempsent_idx'] = tempsent_idx
                Add_Flag=False
                doc_sentidx_key=docid+"_"+str(tempsent_idx)
                for sent_i,sent_pos in enumerate(sent_pos_list):
                    if sent_pos[0]<=start_idx<=end_idx<=sent_pos[1]:
                        doctempsentidx_sent_dict[doc_sentidx_key]=sent_list[sent_i].text
                        Add_Flag=True
                if not Add_Flag:
                    for sent_i,sent_pos in enumerate(sent_pos_list):
                        if sent_pos[0]<=start_idx<=sent_pos[1]:
                            sent_start_index=sent_i
                        if sent_pos[0]<=end_idx<=sent_pos[1]:
                            sent_end_index=sent_i
                            break
                    sent_t=""
                    for sent_spacy in sent_list[sent_start_index:sent_end_index+1]:
                        sent_t=sent_t+sent_spacy.text
                    doctempsentidx_sent_dict[doc_sentidx_key]=sent_t
    return doc_temp_dict,doctempsentidx_sent_dict

def form_match(time_text,tag=""):
    form1="\d{4}-\d{2}-\d{2}"
    form2="\d{4}-\d{2}"
    form3="\d{4}"
    if re.match(form1,time_text):
        time_text=re.match(form1,time_text).group()
    elif re.match(form2,time_text):
        if tag=="begin":
            time_text=re.match(form2,time_text).group()+"-01"
        if tag=="end":
            year, month = int(re.match(form2,time_text).group()[:4]), int(re.match(form2,time_text).group()[5:7])
            try:
                time_text=str(pd.date_range('{}-{}'.format(year, month), periods=1, freq='M')[0])[:10]
                #print(time_text)
            except:
                time_text=""
    elif re.match(form3,time_text):
        time_text=re.match(form3,time_text).group()
        if tag=="begin":
            time_text=re.match(form3,time_text).group()+"-01-01"
        if tag=="end":
            time_text=re.match(form3,time_text).group()+"-12-31"
    else:
        time_text=""
    return time_text

def sutime_dict_to_time_scope(sutime_dict_list):
    time_scope_list=[]
    for i,sutime_dict in enumerate(sutime_dict_list):
        if sutime_dict['type']=="DURATION":
            if isinstance(sutime_dict["value"],dict):
                if len(sutime_dict["value"])==0:
                    begin=""
                    end=""
                    time_scope_list.append([begin,end,sutime_dict])
                else:
                    if "begin" not in sutime_dict["value"].keys():
                        if re.match(".*from \d{4} to \d{4}",sutime_dict["text"].lower()):
                            text=re.search("from \d{4} to \d{4}",sutime_dict["text"].lower()).group()
                            sutime_dict2=sutime.parse(text)[0]
                            sutime_dict2["text"]=sutime_dict["text"]
                            sutime_dict=sutime_dict2
                    if "begin" not in sutime_dict["value"].keys() or "end" not in sutime_dict["value"].keys():
                        begin=""
                        end=""
                    else:
                        begin=sutime_dict["value"]['begin']
                        end=sutime_dict["value"]['end']
                        #form match
                        begin=form_match(begin,"begin")
                        end=form_match(end,"end")
                    time_scope_list.append([begin,end,sutime_dict])
            elif sutime_dict["value"][0]=="P":
                begin=""
                end=""
                time_scope_list.append([begin,end,sutime_dict])
            else:
                begin=""
                end=""
                time_scope_list.append([begin,end,sutime_dict])
        
        if sutime_dict['type']=="SET":
            try:
                if sutime_dict["value"][0]=="P" or sutime_dict["value"][:4]=="XXXX":
                    begin=""
                    end=""
                    time_scope_list.append([begin,end,sutime_dict])
                elif re.match("\d{4}",sutime_dict["value"]):
                    begin=sutime_dict["value"]
                    end=sutime_dict["value"]
                    begin=form_match(begin,"begin")
                    end=form_match(end,"end")
                    time_scope_list.append([begin,end,sutime_dict])
                else:
                    raise
            except:
                begin=""
                end=""
                time_scope_list.append([begin,end,sutime_dict])
        
        if sutime_dict['type']=="TIME":
            begin=sutime_dict["value"]
            end=sutime_dict["value"]
            #form match
            begin=form_match(begin,"begin")
            end=form_match(end,"end")
            time_scope_list.append([begin,end,sutime_dict])
        try:
            if sutime_dict['type']=="DATE":
                begin=sutime_dict["value"]
                end=sutime_dict["value"]
                #form match
                begin=form_match(begin,"begin")
                end=form_match(end,"end")
                time_scope_list.append([begin,end,sutime_dict])
        except:
            begin=""
            end=""
            time_scope_list.append([begin,end,sutime_dict])
            
    return time_scope_list

def normalize_time_scope_list(time_scope_list):
    normalized_time_scope_list=[]
    for time_scope in time_scope_list:
        if time_scope[0]==time_scope[1]=="":
            continue
        else:
            normalized_time_scope_list.append(time_scope)
    return normalized_time_scope_list

def return_cont_temp_list(time_scope_list):
    cont_temp_list=[]
    for time_scope in time_scope_list:
        begin_date=""
        end_date=""
        if time_scope[0]!="":
            begin_date=time_scope[0][:4]+time_scope[0][5:7]+time_scope[0][8:10]
        if time_scope[1]!="":
            end_date=time_scope[1][:4]+time_scope[1][5:7]+time_scope[1][8:10]
        cont_temp_list.append([begin_date,end_date])
    return cont_temp_list

def select_good_records(time_scope_list,sutime_dict_list):
    select_sutime_dict_list=[]
    select_time_scope_list=[]
    for i,time_scope in enumerate(time_scope_list):
        if time_scope[0]==time_scope[1]=="":
            continue
        else:
            select_sutime_dict_list.append(sutime_dict_list[i])
            select_time_scope_list.append(time_scope)
    return select_sutime_dict_list,select_time_scope_list

def return_filteredANDnormalized_tempinfor_dict(doc_temp_dict,doctempsentidx_sent_dict):
    filtered_doc_temp_dict=dict()
    doc_NormalizedTemp_dict=dict()
    for doc_idx,sutime_dict_list in doc_temp_dict.items():
        if len(sutime_dict_list)==0:
            continue
        for tempsent_idx,sutime_dict in enumerate(sutime_dict_list):
            sutime_dict["tempsent_idx"] = tempsent_idx
        raw_time_scope_list=sutime_dict_to_time_scope(sutime_dict_list)
        assert len(raw_time_scope_list)==len(sutime_dict_list)
        select_sutime_dict_list,select_time_scope_list=select_good_records(raw_time_scope_list,sutime_dict_list)
        filtered_doc_temp_dict[doc_idx]=select_sutime_dict_list
        normalized_time_scope_list=normalize_time_scope_list(select_time_scope_list)
        cont_temp_list=return_cont_temp_list(normalized_time_scope_list)
        doc_NormalizedTemp_dict[doc_idx]=cont_temp_list
    
    filtered_doctempsentidx_sent_dict = dict()
    for doc_idx,sutime_dict_list in filtered_doc_temp_dict.items():
        for sutime_dict in sutime_dict_list:
            select_tempsentidx = f"{doc_idx}_{sutime_dict['tempsent_idx']}"
            filtered_doctempsentidx_sent_dict[select_tempsentidx] = doctempsentidx_sent_dict[select_tempsentidx]
    return filtered_doc_temp_dict, doc_NormalizedTemp_dict, filtered_doctempsentidx_sent_dict

def save_tempinfor():
    event_docinfor_data = pickle.load(open("../data/event_data_with_docinfor.pickle", 'rb'))
    docid_doctext_dict = pickle.load(open("../data/docid_doctext_dict.pickle", 'rb'))
    doc_temp_dict,doctempsentidx_sent_dict = basictempinfor_fun(event_docinfor_data,docid_doctext_dict)
    filtered_doc_temp_dict, doc_normalizedtemp_dict, filtered_doctempsentidx_sent_dict = return_filteredANDnormalized_tempinfor_dict(doc_temp_dict,doctempsentidx_sent_dict)
    pickle.dump(filtered_doc_temp_dict, open("../data/filtered_doc_temp_dict.pickle", "wb"))
    pickle.dump(filtered_doctempsentidx_sent_dict, open("../data/filtered_doctempsentidx_sent_dict.pickle", "wb"))
    pickle.dump(doc_normalizedtemp_dict, open("../data/doc_normalizedtemp_dict.pickle", "wb"))

if __name__ == "__main__":
    save_tempinfor()
