

import numpy as np
import sys
from network import NeuralNet

sys.path.insert(1,"../")
from new_util import DataLoader
from preprocess import Preprocess
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


from torch import nn
import torch.nn.functional as F
import torch

#change this with a proper log

dataloader = DataLoader()

print("loading data...")
head,stances,body = dataloader.dataset()

assert len(head) == len(stances) == len(body)


def encoder(stance):
    if stance == "agree":
        return 0
    elif stance == "disagree":
        return 1
    elif stance == "discuss":
        return 2
    elif stance == "unrelated":
        return 3
    else:
        print(stance)
        raise

target = [encoder(i) for i in stances]

try:
    pre = Preprocess()   # Glove embedding http://nlp.stanford.edu/data/glove.6B.zip
    print("parsing...")
    pre.parse()  
except:
    print("download the file from http://nlp.stanford.edu/data/glove.6B.zip")

def representation(data):
    pre_data = []
    for i in data:
        words = []
        
        sep = i.split(" ")
        for s in sep:
            try:
                
                emb = pre.embeddings_dict[s.lower().replace("'","").replace(",","").replace("$15/","")]
                words.append(emb)

            except KeyError:
                pass
                #head_words.append()
        if words != []:
            
            pre_data.append(np.array(words))
   
    return pre_data 



def padding(input,type):
    
    head_max = 200 
    body_max = 1000 
    
    input = input.squeeze()
  
    
    if type == "head":
        
        fixed_pad = head_max  - input.shape[0]
        result = F.pad(input=input,pad=(0,0,0,fixed_pad), mode='constant', value=0)
    elif type == "body":
        
        fixed_pad = body_max  - input.shape[0]
        result = F.pad(input=input,pad=(0,0,0,fixed_pad), mode='constant', value=0)
    else:
        raise
        
    return result

def pos_enc(word_emb,pos):
    
    word_pos,word_emb = [],np.array([i for i in word_emb],dtype=np.float64)
        
    for i in range(len(word_emb)):
        k = 2*i / len(word_emb)
        
        if i%2 == 0:
            word_pos.append(np.sin(pos/(100**k)))
        else:
            word_pos.append(np.cos(pos/(100**k)))
            
   
    return  word_emb +word_pos
    


net = NeuralNet(4)

creatrion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())


torch.save(net,"stance.h5")


exit()

class dataset(torch.utils.data.Dataset):
    def __init__(self):
        
        self.head = head
        self.body = body
        self.target = target
    
    def __len__(self):
        return len(self.head)

    def pipeline(self,idx):

        represent_head = torch.tensor(representation(self.head[idx]))
        represent_body = torch.tensor(representation(self.body[idx]))


        padding_head = padding(represent_head,"head")
        padding_body = padding(represent_body,"body")


        positional_enc_head = np.array([pos_enc(val,i) for i,val in enumerate(padding_head)])
        positional_enc_body = np.array([pos_enc(val,i) for i,val in enumerate(padding_body)])


        return positional_enc_head,positional_enc_body,target[idx]
    
    def __getitem__(self,idx):
        return tuple(self.pipeline(idx))

    
dataset = dataset()
load = torch.utils.data.DataLoader(dataset,batch_size=4)


loss_ = []
data = iter(load)

for i,(h,b,t) in enumerate(data):
    
    optimizer.zero_grad()   
    
    output = net(h,b)
 
    loss = creatrion(output,t) 
    
    loss.backward()
    optimizer.step()  

    loss_.append(loss.item())
    print("Loss ",loss.item())

    if i==50:
        break
