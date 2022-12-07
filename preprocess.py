
import json
import pandas as pd
import glob
import numpy as np


#----get_data----
def get_data(train_path = "./Training_data/train.json"):
    with open(train_path) as json_file:
        train_dict = json.load(json_file) #辞書形式で読み込み
    #train_dict[list(train_dict.keys())[1]]
    return train_dict
    
def get_data_detail(train,path= "./Training_data/Clinical trial json/"):
    statements = []
    primary_section_list = []
    secondary_section_list = []
    primary_evidence_index_list = []
    secondary_evidence_index_list = []
    train_primary_evidence_dict = {}
    train_secondary_evidence_dict = {}
    primary_ctr = None
    secondary_ctr = None
    for i in range(len(train.index)):  
        statements.append(train.iloc[i]["Statement"]) #stamentを取り出す
        #Primary_idを用いてClinical trial jsonのデータを取り出す
        primary_ctr_path = path+train.iloc[i]["Primary_id"]+".json"
        primary_ctr = get_data(primary_ctr_path)
        #Clinical trial jsonのデータからSection_id(Resultsなど)を取り出す
        primary_section = primary_ctr[train.iloc[i]["Section_id"]] #このsectionの内容についてstatementされる
        primary_section_list.append(primary_section)
        #Primary_sectionをさらに細かく見てstatementの根拠が示されているインデックスを取得
        primary_evidence_index = train.iloc[i]["Primary_evidence_index"]
        primary_evidence_index_list.append(primary_evidence_index)
        #複数ある根拠の文を全てevidence_dictに書き出し
        train_evidence = []
        for l in range(len(primary_evidence_index)):
            train_evidence.append(primary_section[primary_evidence_index[l]])
        train_primary_evidence_dict[train.index[i]] = "".join(train_evidence)

        if train.iloc[i]["Type"] == "Comparison":
            secondary_ctr_path = path + train.iloc[i]["Secondary_id"]+".json"
            secondary_ctr = get_data(secondary_ctr_path)
            secondary_section = secondary_ctr[train.iloc[i]["Section_id"]] #このsectionの内容についてstatementされる
            secondary_section_list.append(secondary_section)
            secondary_evidence_index = train.iloc[i]["Secondary_evidence_index"]
            secondary_evidence_index_list.append(secondary_evidence_index)
            train_secondary_evidence = []
            for l in range(len(secondary_evidence_index)):
                train_secondary_evidence.append(secondary_section[secondary_evidence_index[l]])
            train_secondary_evidence_dict[train.index[i]] = "".join(train_secondary_evidence)
            
    return primary_ctr,secondary_ctr,primary_section_list,secondary_section_list,primary_evidence_index_list,secondary_evidence_index_list,train_primary_evidence_dict,train_secondary_evidence_dict

def create_dict(train,train_primary_evidence_dict,train_secondary_evidence_dict):
    dict = {}
    for i in range(len(train.index)):
        if train.iloc[i]["Type"] == "Comparison":
            dict[train.index[i]] = {"Statement": train.iloc[i]["Statement"],
                                "Label": train.iloc[i]["Label"],
                                "Primary_Evidence": train_primary_evidence_dict[train.index[i]],
                                "Secondary_Evidence": train_secondary_evidence_dict[train.index[i]]}
        else:
            dict[train.index[i]] = {"Statement": train.iloc[i]["Statement"],
                                "Label": train.iloc[i]["Label"],
                                "Primary_Evidence": train_primary_evidence_dict[train.index[i]],
                                "Secondary_Evidence": None}
    return dict
if __name__ == "__main__":
    #データの確認
    """
    print(train)
    print(train.info())
    print(train.columns)
    """
    train_dict = get_data()
    train = pd.DataFrame(train_dict) #DataFrame形式にする
    train = train.T
    print("train.jsonのデータ数:",len(train.index))
    print("Statement:",train.iloc[2]["Statement"])
    print("="*100)
    #print(statements)
    print("Clinical Trial ID:",train.iloc[2]["Primary_id"])
    primary_ctr,secondary_ctr,primary_section_list,secondary_section_list,primary_evidence_index_list,secondary_evidence_index_list,train_primary_evidence_dict,train_secondary_evidence_dict= get_data_detail(train)
    print(primary_ctr)
    print("="*100)
    print("Section_id:",train.iloc[2]["Section_id"])
    print("="*100)
    print("primary_section:",len(primary_section_list))
    print("="*100)
    print("primary_evidence_index:",len(primary_evidence_index_list)) #1650
    print("="*100)
    print("train_primary_evidence:",len(train_primary_evidence_dict)) #1650
    print("="*100)
    print("secondary_section:",len(secondary_section_list)) #615
    print("="*100)
    print("secondary_evidence_index:",len(secondary_evidence_index_list)) #615
    print("="*100)
    print("train_secondary_evidence:",len(train_secondary_evidence_dict)) #615

    dict = create_dict(train,train_primary_evidence_dict,train_secondary_evidence_dict)
    df_dict = pd.DataFrame(dict).T
    pd.set_option("display.max_rows",None)
    #print(df_dict)
    #df_dict.to_pickle("dataset.pkl")
    df_dict.to_csv("train_dataset.tsv",sep = '\t')
