import torch
from transformers import BertTokenizer,BertModel,DataCollatorWithPadding
from torch.utils.data import Dataset,TensorDataset,DataLoader
import pandas as pd
import argparse
from tqdm import tqdm
import wandb

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#read_data
def open_file(path):
    read_table = pd.read_table(path,header=0,index_col = 0) #headerにindexを割り振らない
    return read_table
#max_len
def max_len(x,y,tokenizer):
    max_len = 0
    for i in range(len(y)):
        token = tokenizer.tokenize(x[i])
        if max_len < len(token):
            max_len = len(token)
    return max_len

#statementにevidenceを加える
def data_concat(df):
    concat = []
    """
    for i in range(len(df.index)):
        concat.append(df.iloc[i]["Statement"]+df.iloc[i]["Primary_Evidence"]) #max_sent_len 1976となり、BERTに入らない
    """
    return concat

#dataset
class CustomDataset(Dataset):
    def __init__(self,x,y,tokenizer,max_len):
        self.x = x
        self.y = y
        #self.pe = pe
        #self.se = se
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self,index):
        #1650
        text = self.x[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return torch.LongTensor(ids),torch.LongTensor(mask),torch.tensor(self.y[index])

def dataset(df,y,max_len):
    custom_dataset = CustomDataset(df["Statement"].values,y,tokenizer,max_len = max_len)
    ids = torch.zeros(len(custom_dataset),max_len)
    mask = torch.zeros(len(custom_dataset),max_len)
    labels = torch.zeros(len(custom_dataset))
    #print(len(custom_dataset))
    for i in range(len(custom_dataset)):
        ids[i][:] = custom_dataset[i][0]
        mask[i][:] = custom_dataset[i][1]
        labels[i] = custom_dataset[i][2]
    return ids,mask,labels

def dataloader(ids,mask,labels,batch_size):
    #ids,mask,labels --> tensor
    dataset = TensorDataset(ids,mask,labels)
    dataloader = DataLoader(dataset,batch_size = batch_size,shuffle = True)
    return dataloader

#Bert implement
class Bert(torch.nn.Module):
    def __init__(self,output_size,drop_rate):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(p = drop_rate)
        self.linear = torch.nn.Linear(768,output_size) #bertの出力の次元768
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self,ids,mask):
        _,output = self.bert(ids,mask,return_dict = False)
        output = self.dropout(output)
        output = self.linear(output) #(30,1)
        output = output.squeeze()
        output = self.sigmoid(output)
        return output

#損失関数の定義
def loss_function():
    loss = torch.nn.BCELoss()
    return loss

#学習
def train_step(model,loss,epoch,device,w_and_b,confirm,patience):
    num = 0
    F_max = 0
    for epo in range(epoch):
        train_losses = torch.zeros(len(train_dataloader))
        dev_losses = torch.zeros(len(dev_dataloader))
        F_scores = torch.zeros(len(train_dataloader))
        accuracies = torch.zeros(len(train_dataloader))
        dev_F_scores = torch.zeros(len(dev_dataloader))
        dev_accuracies = torch.zeros(len(dev_dataloader))
        model.train()
        for i,(train_x,train_mask,train_y) in enumerate(tqdm(train_dataloader)):
            #train_x(batch_size,max_len)
            train_x,train_mask,train_y = train_x.to(device).long(),train_mask.to(device).long(),train_y.to(device).float()
            optimizer.zero_grad()
            y_tilde = model(train_x,train_mask).float()
            train_loss = loss(y_tilde,train_y)
            train_loss.backward()
            optimizer.step()

            pred_list = list(torch.round(y_tilde))

            y_list = list(train_y)
            F_score,accuracy,recall,precision = evaluate(y_list,pred_list)
            train_losses[i] = train_loss
            F_scores[i] = F_score
            accuracies[i] = accuracy

            if confirm == True:
                correct = 0
                error = 0
                for k,l in enumerate(y_list):
                    if int(l) == 1:
                        correct += 1
                    else:
                        error += 1
                if i % 55 == 0:
                    print("correct + error = batch_size ",correct+error)
                    print("correct = ",correct)
                    print("error = ",error)
                    print("correct/(correct+error) = ",correct/(correct+error))
            
        model.eval()
        with torch.no_grad():
            for i,(dev_x,dev_mask,dev_y) in enumerate(tqdm(dev_dataloader)):
                dev_x,dev_mask,dev_y = dev_x.to(device).long(),dev_mask.to(device).long(),dev_y.to(device).float()
                y_tilde_dev = model(dev_x,dev_mask).float() #予測
                dev_loss = loss(y_tilde_dev,dev_y) #loss
                dev_pred_list = list(torch.round(y_tilde_dev)) #0~1に変換
                dev_y_list = list(dev_y)
                dev_F_score,dev_accuracy,dev_recall,dev_precision = evaluate(dev_y_list,dev_pred_list) #評価
                dev_losses[i] = dev_loss
                dev_F_scores[i] = dev_F_score
                dev_accuracies[i] = dev_accuracy

        train_mean_loss = torch.mean(train_losses)
        F_mean_score = torch.mean(F_scores)
        accuracy_mean = torch.mean(accuracies)
        dev_mean_loss = torch.mean(dev_losses)
        dev_F_mean_score = torch.mean(dev_F_scores)
        dev_accuracy_mean = torch.mean(dev_accuracies)
        
        print("train:: epo = {},train_loss = {},F_score = {},accuracy = {}".format(epo,train_mean_loss,F_mean_score,accuracy_mean))
        print("dev:: epo = {},dev_loss = {},F_score = {},accuracy = {}".format(epo,dev_mean_loss,dev_F_mean_score,dev_accuracy_mean))
        if w_and_b:
            wandb.log({
                "epoch:loss/train":train_mean_loss,
                "epoch: F_score/train":F_mean_score,
                "epoch: Accuracy/trian":accuracy_mean,
                "epoch: loss/dev":dev_mean_loss,
                "epoch: F_score/dev":dev_F_mean_score,
                "epoch: accuracy/dev":dev_accuracy_mean})
        
        if dev_F_mean_score > F_max:
            F_max = dev_F_mean_score
            num = 0
        else:
            num += 1
        
        if num == patience:
            print("Early Stopping")
            break

    torch.save(model.state_dict(),"./bert_model_statement_only_add_dev")
    return dev_F_mean_score,dev_accuracy_mean

def evaluate(y_list,pred_list):
    tp,fp,fn,tn = 1e-14,1e-14,1e-14,1e-14,
    for y,pred in zip(y_list,pred_list):
        if y ==1 and pred == 1:
            tp += 1
        elif y == 1 and pred == 0:
            fn += 1
        elif y == 0 and pred == 1:
            fp += 1
        else:
            tn += 1
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    F_score = 2*recall*precision/(recall+precision)
    return F_score,accuracy,recall,precision

if __name__ == "__main__":

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    parser = argparse.ArgumentParser(description="hyperparameter")
    parser.add_argument("--lr",type = float,)
    parser.add_argument("--device",type = str,default= False)
    parser.add_argument("--epoch",type = int)
    parser.add_argument("--wandb",type = str,default=False)
    parser.add_argument("--patience",type = int,help = "early stopping")
    parser.add_argument("--confirm",type = str,default = False)
    args = parser.parse_args()
    lr = args.lr
    epoch = args.epoch
    w_and_b = args.wandb
    patience = args.patience
    confirm = args.confirm
    #wandb 
    if w_and_b:
        wandb.init(project = "NLI4CT")
        wandb.config = {
            "lr":lr,
            "epoch":epoch,
            "patience":patience,
        }
    #gpu設定
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.device) else "cpu")

    df = open_file("train_dataset.tsv")
    df_train,df_dev = train_test_split(df,test_size = 0.2) #train: 1320 dev: 330
    #print(df)
    #正解ラベルをEntailment-->1 Contradiction -->0
    y_train = pd.get_dummies(df_train,columns=["Label"])["Label_Entailment"].values #list形式
    y_dev = pd.get_dummies(df_dev,columns=["Label"])["Label_Entailment"].values

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #max_len
    train_max_len = max_len(df_train["Statement"].values,y_train,tokenizer)
    dev_max_len = max_len(df_dev["Statement"].values,y_dev,tokenizer)

    print("train : max_len",train_max_len)
    print("dev : max_len",dev_max_len)

    #train_custom_dataset = CustomDataset(df_train["Statement"].values,y_train,tokenizer,max_len = train_max_len)
    #dev_custom_dataset = CustomDataset(df_dev["Statement"].values,y_dev,tokenizer,max_len = dev_max_len)
    train_ids,train_mask,train_labels = dataset(df_train,y_train,train_max_len)
    dev_ids,dev_mask,dev_labels = dataset(df_dev,y_dev,dev_max_len)
    train_dataloader = dataloader(train_ids,train_mask,train_labels,batch_size=30)
    dev_dataloader = dataloader(dev_ids,dev_mask,dev_labels,batch_size=30)
    #token = tokenizer.convert_ids_to_tokens(train_ids[0]) #id->tokenに変換してくれる
    #print(token)
    model = Bert(1,0.3)
    model.to(device)
    loss = loss_function()
    #最適化関数
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
    
    train_step(model,loss,epoch,device,w_and_b,confirm,patience)