import math
import argparse
from argparse import Namespace
import torch.nn as nn
from train_utils import *
from datasets import Event_Seq_Dataset, generate_batches
from torch.nn import TransformerEncoder, TransformerEncoderLayer
outputsize_dict={"Day":7475,"Week":1069,"Month":246,"Year":21}
#Number of days/weeks/months/years of the NYT corpus
set_seed_everywhere(seed=0)

class CNN_Transformer_Model(nn.Module):
    def __init__(self,Trans_model_args):
        super(CNN_Transformer_Model,self).__init__()
        in_channels=Trans_model_args.in_channels
        out_channels=Trans_model_args.out_channels
        kernel_size=Trans_model_args.kernel_size
        stride=Trans_model_args.stride
        padding=Trans_model_args.padding
        nhead=Trans_model_args.nhead
        nhid=Trans_model_args.nhid 
        dropout=Trans_model_args.dropout
        nlayers=Trans_model_args.nlayers
        self.model_output_size = Trans_model_args.output_dim
        self.temp_granularity = Trans_model_args.temp_granularity

        self.convs_list2=nn.Sequential(
        nn.Conv1d(in_channels,out_channels,kernel_size,stride,kernel_size//2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU())
        self.convs_list2_2=nn.Sequential(
        nn.Conv1d(out_channels,out_channels*2,kernel_size,stride,kernel_size//2), 
        nn.BatchNorm1d(out_channels*2),
        nn.ReLU())
        self.pos_encoder=PositionalEncoding(out_channels*2,self.temp_granularity,dropout)
        encoder_layers = TransformerEncoderLayer(out_channels*2, nhead, nhid, dropout) 
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
    def forward(self, x_in, apply_softmax=False):
        x_in=x_in.permute(0,2,1)
        cnn_output2 = self.convs_list2(x_in)
        if cnn_output2.shape[-1]!=self.model_output_size:
            cnn_output2 = cnn_output2[:,:,:-1]
        cnn_output2 = self.convs_list2_2(cnn_output2)
        if cnn_output2.shape[-1]!=self.model_output_size:
            cnn_output2 = cnn_output2[:,:,:-1]
        cnn_output=cnn_output2
        cnn_output=cnn_output.permute(2,0,1) 
        cnn_output = self.pos_encoder(cnn_output)
        trans_output = self.transformer_encoder(cnn_output)
        trans_output=trans_output.permute(1,0,2).mean(dim=2)  
        y_out=trans_output
        return y_out

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,temp_granularity,dropout=0.1,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        if temp_granularity=="Day":
            max_len=8000
        pe=torch.zeros(max_len,d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self,x):
        x=x+self.pe[:x.size(0),:]
        return self.dropout(x)

def train_process(dataset_args, model_args, dataset, model):
    model = model.to(dataset_args.device)
    optimizer = optim.Adam(model.parameters(), lr=model_args.learning_rate, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)
    train_state = make_train_state(model_args,dataset_args.model_state_file)
    loss_func = loss_function(model_args)
    epoch_bar = tqdm(desc='training routine', total=dataset_args.num_epochs, position=0)
    dataset.set_split('train')
    train_bar = tqdm(desc='split=train',total=dataset.get_num_batches(dataset_args.batch_size),position=1,leave=True)
    dataset.set_split('eval')
    val_bar = tqdm(desc='split=val',total=dataset.get_num_batches(dataset_args.batch_size),position=1,leave=True)
    train_column_name=dataset_args.train_column_name
    try:
        for epoch_index in range(dataset_args.num_epochs):
            train_state['epoch_index'] = epoch_index
            dataset.set_split('train')
            batch_generator = generate_batches(dataset, batch_size=dataset_args.batch_size, device=dataset_args.device)
            running_loss = 0.0
            running_acc = 0.0
            running_abs_loss = 0.0
            model.train()
            for batch_index, batch_dict in enumerate(batch_generator):
                # step 1. zero the gradients
                optimizer.zero_grad()
                # step 2. compute the output
                y_pred = model(x_in=batch_dict[train_column_name].float())
                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['event_dateidx'].long())
                # step 4. use loss to produce gradients
                loss.backward()
                # step 5. use optimizer to take gradient step
                optimizer.step()
                # compute the  running loss and running accuracy
                running_loss += (loss.item() - running_loss) / (batch_index + 1)
                acc_t = compute_accuracy(y_pred, batch_dict['event_dateidx'].long())
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                abs_loss=absolute_error(y_pred, batch_dict['event_dateidx'].long())
                running_abs_loss += (abs_loss - running_abs_loss) / (batch_index + 1)
                train_bar.set_postfix(loss=running_loss,acc=running_acc,abs_error=running_abs_loss,epoch=epoch_index)
                train_bar.update()
            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)
            dataset.set_split('eval')
            batch_generator = generate_batches(dataset,batch_size=dataset_args.batch_size,device=dataset_args.device)
            running_loss = 0.0
            running_acc = 0.0
            running_abs_loss = 0.0
            model.eval()
            for batch_index, batch_dict in enumerate(batch_generator):
                y_pred = model(x_in=batch_dict[train_column_name].float())
                loss = loss_func(y_pred, batch_dict['event_dateidx'].long())
                running_loss += (loss.item() - running_loss) / (batch_index + 1)
                acc_t = compute_accuracy(y_pred, batch_dict['event_dateidx'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                abs_loss=absolute_error(y_pred, batch_dict['event_dateidx'].long())
                running_abs_loss += (abs_loss - running_abs_loss) / (batch_index + 1)
                val_bar.set_postfix(loss=running_loss, acc=running_acc, abs_error=running_abs_loss, epoch=epoch_index)
                val_bar.update()
            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)
            train_state = update_train_state(model_args=model_args, model=model,train_state=train_state)
            scheduler.step(train_state['val_loss'][-1])
            if train_state['stop_early']:
                break
            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()
    except KeyboardInterrupt:
        print("Exiting loop")
    return model,train_state

def train_model(temp_granularity):
    model_output_size=outputsize_dict[temp_granularity]
    date_to_idx_dict=return_auxiliary_data(temp_granularity)
    dataset_args = Namespace(
        pub_topN=50,
        StandardScaler_Flag=True,
        train_column_name="Multi_timeseries",
        cuda=True,
        batch_size=64,
        num_epochs=100,
        test_batch_size=4,
        model_state_file="CNN_Trans_"+temp_granularity+".pth",
        save_dir="./data/models",
        expand_filepaths_to_save_dir=True,
        reload_from_files=False,
        delete_previous_model = True
    )
    model_args = Namespace(
        in_channels=4,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=0,
        nhead=4,
        nhid=200,
        dropout=0.2,
        nlayers=3,
        output_dim=model_output_size,
        temp_granularity=temp_granularity,
        loss="cross_entropy",
        early_stopping_criteria=5,
        learning_rate=0.001,
        dropout_p=0.0
    )
    dataset_args=process_dataset_args(dataset_args,temp_granularity,model_output_size)
    dataset = Event_Seq_Dataset.load_dataset(dataset_args.data_reader_param, date_to_idx_dict)
    model = CNN_Transformer_Model(model_args)
    model,train_state_dict=train_process(dataset_args, model_args, dataset, model)

def test_model(temp_granularity):
    model_output_size=outputsize_dict[temp_granularity]
    date_to_idx_dict=return_auxiliary_data(temp_granularity)
    dataset_args = Namespace(
        pub_topN=50,
        StandardScaler_Flag=True,
        train_column_name="Multi_timeseries",
        cuda=True,
        batch_size=64,
        num_epochs=100,
        test_batch_size=4,
        model_state_file="CNN_Trans_"+temp_granularity+".pth",
        save_dir="./data/models",
        expand_filepaths_to_save_dir=True,
        reload_from_files=False,
        delete_previous_model = False
    )
    model_args = Namespace(
        in_channels=4,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=0,
        nhead=4,
        nhid=200,
        dropout=0.2,
        nlayers=3,
        output_dim=model_output_size,
        temp_granularity=temp_granularity,
        loss="cross_entropy",
        early_stopping_criteria=5,
        learning_rate=0.001,
        dropout_p=0.0
    )
    dataset_args=process_dataset_args(dataset_args,temp_granularity,model_output_size)
    dataset = Event_Seq_Dataset.load_dataset(dataset_args.data_reader_param, date_to_idx_dict)

    if os.path.isfile(dataset_args.model_state_file):
        model = CNN_Transformer_Model(model_args)
        loss_func = loss_function(model_args)
        dataset.set_split('test')
        batch_generator = generate_batches(dataset,batch_size=dataset_args.test_batch_size,device=dataset_args.device)
        # compute the loss & accuracy on the test set using the best available model
        model.load_state_dict(torch.load(dataset_args.model_state_file))
        model = model.to(dataset_args.device)
        train_column_name= dataset_args.train_column_name
        running_loss = 0.0
        running_acc = 0.0
        running_abs_loss = 0.0
        model.eval()
        correct_records_list=[]
        incorrect_records_list=[]
        incorrect_predict_list=[]
        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            y_pred =  model(batch_dict[train_column_name].float())
            # compute the loss
            loss = loss_func(y_pred, batch_dict['event_dateidx'].long())
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['event_dateidx'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            abs_loss=absolute_error(y_pred, batch_dict['event_dateidx'].long())
            running_abs_loss += (abs_loss - running_abs_loss) / (batch_index + 1)
            y_true=batch_dict['event_dateidx'].long()
            _, y_pred_indices = y_pred.max(dim=1)
            correct_indices = torch.eq(y_pred_indices, y_true).float()
            for i,val in enumerate(correct_indices.int()):
                records=batch_dict["event_idx"][i].item()
                if val==1:
                    correct_records_list.append(records)
                else:
                    incorrect_records_list.append(records)
                    incorrect_predict_list.append(y_pred_indices[i].item())
        print("Test Accuracy:{}; MAE:{}".format(running_acc,running_abs_loss))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temp_granularity",
        default="Year",
        type=str,
        help="Choose granularity.",
        choices=['Year', 'Month', 'Week', 'Day']
    )
    args = parser.parse_args()
    print(f"Train the model with temp_granularity: {args.temp_granularity}")
    train_model(args.temp_granularity)
    print("---------------------------------------------------------")
    print("Test the trained model.")
    test_model(args.temp_granularity)

if __name__ == "__main__":
    main()
