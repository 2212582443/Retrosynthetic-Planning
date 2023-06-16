import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToMolBlock
from rdkit.Chem import SDWriter,AllChem
from torch.utils.data import Dataset, TensorDataset
from rdchiral.template_extractor import extract_from_reaction
import torch
import numpy as np


#将反应式拆分为单个产物反应，返回拆分后的列表
def split_reaction(reaction):
    rct, prd = reaction.split('>>')
    prd_mols = [x for x in prd.split('.')]
    
    reactions = []
    for p in prd_mols:
        reaction = [rct].copy()
        reaction.append(p)
        reactions.append(reaction)
    
    return reactions

#根据smiles形式的反应式得到template
def get_template(reactants,products):
    inputRec = {'_id':None, 'reactants':reactants, 'products':products}
    ans = extract_from_reaction(inputRec)
    if 'reaction_smarts' in ans.keys():
        return ans['reaction_smarts']
    else:
        return None
    

#将smiles形式的产物转化为摩根指纹向量
def get_mfp(product):
    mol = Chem.MolFromSmiles(product)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(),dtype = bool)
    arr[onbits] = 1
    return arr

#直接提取模板
def preprocess_data(path,name):
    input_path = path+'raw_'+name+'.csv'
    output_path = path+name+'_dataset.pt'
    df = pd.read_csv(input_path)
    features = []
    labels = []
    num_none = 0

    for i, row in df.iterrows():
        reactions = split_reaction(row['reactants>reagents>production'])
        for r in reactions:
            template = get_template(r[0],r[1])
            mfp = get_mfp(r[1])
            if template == None:
                num_none+=1
                continue
            features.append(mfp)
            labels.append(template)

    dataset = [(feature,label) for label, feature in zip(labels, features)]
    print('dataset size:', len(dataset))
    # 保存训练、验证和测试数据集
    torch.save(dataset,output_path)

#筛选可用模板
def get_valid_labels(input_path):
    df = pd.read_csv(input_path)
    dic = dict()

    for i, row in df.iterrows():
        reactions = split_reaction(row['reactants>reagents>production'])
        for r in reactions:
            template = get_template(r[0],r[1])
            if template == None:
                continue
            if dic.get(template) is None:
                dic[template]=1
            else:
                dic[template]+=1
    
    labels_num=0
    samples_num=0
    valid_num=0
    valid_labels=[]
    for key,value in dic.items():
        samples_num+=value
        if value>=100:
            labels_num+=1
            valid_num+=value
            #print(value)
            valid_labels.append(key)
    print(f"valid labels has {labels_num}")
    print(f"valid samples is {valid_num}/{samples_num}")

    return valid_labels

#筛选数据集
def preprocess_data_filter(path,name,valid_labels):
    input_path = path+'raw_'+name+'.csv'
    output_path = path+name+'_filter_dataset.pt'
    df = pd.read_csv(input_path)
    features = []
    labels = []
    num_none = 0

    for i, row in df.iterrows():
        reactions = split_reaction(row['reactants>reagents>production'])
        for r in reactions:
            template = get_template(r[0],r[1])
            mfp = get_mfp(r[1])
            if template == None:
                num_none+=1
                continue
            if template in valid_labels:
                features.append(mfp)
                labels.append(template)

    dataset = [(feature,label) for label, feature in zip(labels, features)]
    print('filtered dataset size:', len(dataset))
    # 保存训练、验证和测试数据集
    torch.save(dataset,output_path)


    



if __name__ == '__main__':
    path = 'schneider50k/'

    for name in 'train','val','test':
        print(name+' dataset:')
        preprocess_data(path,name)

    valid_labels = get_valid_labels(path+'raw_train.csv')
    for name in 'train','val','test':
        print(name+' filter dataset:')
        preprocess_data_filter(path,name,valid_labels)
