import os
import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import torch.optim as optim
from torch.nn import functional as F

def set_seed_everywhere(seed):    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

def time_count_dict_fun(model_output_size):
    time_count_dict={}
    for idx in range(model_output_size):
        time_count_dict[idx]=0
    return time_count_dict

def return_auxiliary_data(temp_granularity):
    date_prediction_list=[]
    for year_i in range(1987,2008):
        start_date=datetime(year_i,1,1)
        if year_i==2007:
            end_date=datetime(year_i,6,19)
        else:
            end_date=datetime(year_i,12,31)
        d=start_date
        dates=[start_date]
        while d < end_date:
            d += timedelta(days=1)
            dates.append(d)
        for date in dates:
            date_prediction_list.append(date.strftime("%Y-%m-%d"))
    if temp_granularity=="Year":
        date_to_idx_dict=dict()
        for date in date_prediction_list:
            date_to_idx_dict[date]=int(date[:4])-1987
        return date_to_idx_dict
    if temp_granularity=="Month":
        date_to_idx_dict=dict()
        for date in date_prediction_list:
            month_index=(int(date[:4])-1987)*12+int(date[5:7])-1
            date_to_idx_dict[date]=month_index
        return date_to_idx_dict
    if temp_granularity=="Day":
        date_to_idx_dict=dict()
        for idx,date in enumerate(date_prediction_list):
            date_time=date
            date_to_idx_dict[date_time]=idx
        return date_to_idx_dict
    if temp_granularity=="Week":
        date_to_idx_dict=dict()
        week_idx=0
        for date_infor in ['1987-01-01', '1987-01-02', '1987-01-03', '1987-01-04']:
            date_to_idx_dict[date_infor]=week_idx
        week_idx+=1
        date_format = "%Y-%m-%d"
        next_date='1987-01-04'
        new_week_last_date=(datetime.strptime('1987-01-04', date_format)+timedelta(days=7)).strftime("%Y-%m-%d")
        last_date="2007-06-19"
        while (next_date!=last_date):
            next_dateime=datetime.strptime(next_date, date_format)+timedelta(days=1)
            next_date=next_dateime.strftime("%Y-%m-%d")
            if next_date!=new_week_last_date:
                date_to_idx_dict[next_date]=week_idx
            else:
                new_week_last_date=(datetime.strptime(next_date, date_format)+timedelta(days=7)).strftime("%Y-%m-%d")
                date_to_idx_dict[next_date]=week_idx
                week_idx+=1
        return date_to_idx_dict

def process_dataset_args(Model_dataset_args,temp_granularity,model_output_size):
    Model_dataset_args.data_reader_param=[Model_dataset_args.pub_topN,Model_dataset_args.StandardScaler_Flag,temp_granularity,model_output_size]
    if not torch.cuda.is_available():
        Model_dataset_args.cuda = False
    Model_dataset_args.device = torch.device("cuda" if Model_dataset_args.cuda else "cpu")
    print("Using CUDA: {}".format(Model_dataset_args.cuda))
    if Model_dataset_args.expand_filepaths_to_save_dir:
        Model_dataset_args.model_state_file = os.path.join(Model_dataset_args.save_dir,Model_dataset_args.model_state_file)
        print("Expanded filepaths: ")
        print("\t{}".format(Model_dataset_args.model_state_file))
    if os.path.exists(Model_dataset_args.model_state_file):
        if Model_dataset_args.delete_previous_model:
            print("Delete Saved Former Model.")
            os.remove(Model_dataset_args.model_state_file)
    if not os.path.exists(Model_dataset_args.save_dir):
        os.makedirs(Model_dataset_args.save_dir)
    return Model_dataset_args        

def compute_accuracy(y_pred, y_true):
    _, y_pred_indices = torch.max(y_pred, 1)
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    n_correct = correct_indices.sum().item()
    n_sum = len(correct_indices)
    return n_correct / n_sum * 100

def absolute_error(y_pred, y_true):
    _, y_pred_indices = torch.max(y_pred, 1)
    abs_error=abs((y_pred_indices-y_true)).sum().item()/len(y_pred_indices)
    return abs_error

def make_train_state(model_args,model_state_file):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': model_args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': model_state_file}

def update_train_state(model_args, model, train_state):
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False
    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]
        # If loss worsened
        if loss_t >= loss_tm1:
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])
                train_state['early_stopping_best_val'] = loss_t
            train_state['early_stopping_step'] = 0
        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= model_args.early_stopping_criteria
    train_state['stop_early']=False
    return train_state

def loss_function(model_args):
    if model_args.loss=="cross_entropy":
        loss_func = F.cross_entropy
    return loss_func

