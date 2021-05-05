import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset,DataLoader
from datetime import datetime, timedelta
from train_utils import *

def return_NN_PubTrainingData(event_temp_data, pub_topN, StandardScaler_Flag, temp_granularity, model_output_size):
    doc_TempInfor_dict=pickle.load(open("./data/input_data/doc_tempinfor_input.pickle", "rb"))
    temp_seqidx_data=[]
    date_to_idx_dict=return_auxiliary_data(temp_granularity)
    for idx, event_temp_record in enumerate(event_temp_data):
        event_id=event_temp_record[0]
        event_text=event_temp_record[1]
        predicted_idx=event_temp_record[2]
        pub_list=event_temp_record[3][:pub_topN]
        pub_list=[date_to_idx_dict[pub_value] for pub_value in pub_list]
        cont_list=event_temp_record[4][:pub_topN]
        doc_idx_list=event_temp_record[5][:pub_topN]
        split=event_temp_record[6]
        max_doc_rel=cont_list[0]
        pub_month_count_dict=time_count_dict_fun(model_output_size)
        cont_month_count_dict=time_count_dict_fun(model_output_size)
        for i,date_idx in enumerate(pub_list):
            pub_month_count_dict[date_idx]+=1 
            cont_month_count_dict[date_idx]+=(cont_list[i]/max_doc_rel)
        
        conttemp_month_count_dict=time_count_dict_fun(model_output_size)
        conttempsim_month_count_dict=time_count_dict_fun(model_output_size)
        
        for idx2,event_doc_idx in enumerate(doc_idx_list):
            normalized_temp=[]
            if event_doc_idx in doc_TempInfor_dict:
                for tempinfor_dict in doc_TempInfor_dict[event_doc_idx]:
                    temp_date_list=tempinfor_dict["normalized_temp"]
                    temp_list=[]
                    for t_date in temp_date_list:
                        temp_list.append(t_date[:4]+"-"+t_date[4:6]+"-"+t_date[6:8])
                    normalized_temp.append(temp_list)
                temp_simscore_list=doc_TempInfor_dict[event_doc_idx] 
                
            for temp_i,temp_l in enumerate(normalized_temp):
                start_temp=temp_l[0]
                end_temp=temp_l[1]
                if len(start_temp)==0:
                    start_temp=end_temp
                if len(end_temp)==0:
                    end_temp=start_temp
                if start_temp not in date_to_idx_dict:
                    continue
                if end_temp not in date_to_idx_dict:
                    continue
                begin_temp_index=date_to_idx_dict[start_temp]
                end_temp_index=date_to_idx_dict[end_temp]
                if begin_temp_index>end_temp_index:
                    continue
                if begin_temp_index<0 or begin_temp_index>(model_output_size-1):
                    continue
                if end_temp_index<0 or end_temp_index>(model_output_size-1):
                    continue    
                score=1.0/(end_temp_index-begin_temp_index+1)
                simscore=temp_simscore_list[temp_i]['event_sent_score'][str(event_id)]
                simscore=simscore/(end_temp_index-begin_temp_index+1)
                for temp_index in range(begin_temp_index,end_temp_index+1):
                    if 0<=temp_index<=(model_output_size-1):
                        conttemp_month_count_dict[temp_index]+=score
                        conttempsim_month_count_dict[temp_index]+=simscore

        temp_seqidx_data.append([event_id,event_text,predicted_idx,\
                                 list(pub_month_count_dict.values()),pub_list,\
                                 list(cont_month_count_dict.values()),cont_list,\
                                 list(conttemp_month_count_dict.values()),\
                                 list(conttempsim_month_count_dict.values()),\
                                 split])
    if StandardScaler_Flag==True:
        temp_seqidx_data=StandardScaler_Fun(temp_seqidx_data)
    return temp_seqidx_data

def StandardScaler_Fun(temp_seqidx_data):
    temp_seqidx_columns=["event_idx","event_text","event_dateidx",\
                         "pub_month_count","pub_seq",\
                         "contrel_month_count","contrel_seq",\
                         "conttemp_month_count_dict",'conttempsim_month_count',"split"]
    temp_seqidx_data_pd=pd.DataFrame(temp_seqidx_data,columns=temp_seqidx_columns)
    col_names = ['pub_month_count', 'contrel_month_count','conttemp_month_count_dict','conttempsim_month_count']
    for col_name in col_names:
        features =temp_seqidx_data_pd[temp_seqidx_data_pd["split"]=="train"][col_name]
        features = np.array(features.values.tolist())
        scaler = StandardScaler().fit(features)
        all_features = np.array(temp_seqidx_data_pd[col_name].values.tolist())
        temp_seqidx_data_pd[col_name] = scaler.transform(all_features).tolist()
    train_record=list(temp_seqidx_data_pd[temp_seqidx_data_pd["split"]=="train"].values)
    train_record_label=list(temp_seqidx_data_pd[temp_seqidx_data_pd["split"]=="train"].event_dateidx.values)
    return list(temp_seqidx_data_pd.values)

class Event_Seq_Dataset(Dataset):
    def __init__(self, record_df, date_to_idx_dict):
        self.record_df = record_df
        self.train_df = self.record_df[self.record_df.split=='train']
        self.eval_df = self.record_df[self.record_df.split=='eval']
        self.test_df = self.record_df[self.record_df.split=='test']
        self.train_size = len(self.train_df)
        self.eval_size = len(self.eval_df)
        self.test_size = len(self.test_df)
        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'eval': (self.eval_df, self.eval_size),
                             'test': (self.test_df, self.test_size)}
        self.set_split('train')
        self.date_to_idx_dict = date_to_idx_dict
        print("train size:{}, eval size:{}, test size:{}".format(self.train_size,self.eval_size,self.test_size))
        
    @staticmethod
    def return_labeled_data(data_reader_param):
        with open("./data/input_data/event_docinfor_input.pickle", 'rb') as f:
            event_temp_data = pickle.load(f)
        data_reader_param_copy=data_reader_param.copy()
        data_reader_param_copy.insert(0,event_temp_data)
        event_temp_training_data=return_NN_PubTrainingData(*data_reader_param_copy)
        
        temp_seqidx_columns=["event_idx","event_text","event_dateidx",\
                             "pub_month_count","pub_seq",\
                             "contrel_month_count","contrel_seq",\
                             "conttemp_month_count_dict",'conttempsim_month_count',"split"]
        event_temp_training_pd=pd.DataFrame(event_temp_training_data,columns=temp_seqidx_columns)
        return event_temp_training_pd
    
    @classmethod
    def load_dataset(cls,data_reader_param,date_to_idx_dict):
        record_df = cls.return_labeled_data(data_reader_param)
        return cls(record_df,date_to_idx_dict)

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        event_row = self._target_df.iloc[index]
        event_idx = event_row[0]
        event_dateidx = self.date_to_idx_dict[event_row[2]]
        pub_month_count=np.array(event_row[3]).reshape(-1,1)
        contrel_month_count=np.array(event_row[5]).reshape(-1,1)
        conttemp_month_count=np.array(event_row[7]).reshape(-1,1)
        conttempsim_month_count=np.array(event_row[8]).reshape(-1,1)
        concate_month_count=np.concatenate((pub_month_count,contrel_month_count,conttemp_month_count,conttempsim_month_count),axis=1)
        return {"event_idx": int(event_idx),
                'event_dateidx': int(event_dateidx),
                'Pubdate_timeseries': np.array(pub_month_count,dtype=np.float64),
                'Doc2event_timeseries': np.array(contrel_month_count,dtype=np.float64),
                'Contdate_timeseries': np.array(conttemp_month_count,dtype=np.float64),
                'Sent2event_timeseries': np.array(conttempsim_month_count,dtype=np.float64),
                'Multi_timeseries': np.array(concate_month_count,dtype=np.float64)}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

def generate_batches(dataset, batch_size, shuffle=True,drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,shuffle=shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device=device)
        yield out_data_dict