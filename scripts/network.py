import sys

sys.path.insert(1,"../")
from self_attention import MultiHeadAttention
from torch.utils.tensorboard import SummaryWriter

from torch import nn
import torch.nn.functional as F
import torch


class NeuralNet(nn.Module):
    def __init__(self,batch_size = 1):
        
        super(NeuralNet,self).__init__()
        
        self.attention_head = MultiHeadAttention(in_features=50, head_num=1)
        self.batch_size = batch_size
        self.Q_head = torch.rand(self.batch_size,200,50) 
        self.K_head = torch.rand(self.batch_size,200,50)
        self.V_head = torch.rand(self.batch_size,200,50)
        
        self.attention_body = MultiHeadAttention(in_features=50, head_num=1)
        
        self.Q_body = torch.rand(self.batch_size,1000,50)
        self.K_body = torch.rand(self.batch_size,1000,50)
        self.V_body = torch.rand(self.batch_size,1000,50)
        
        # fully connected net 
                                
        self.fc1 = nn.Linear(60000, 500)   
    
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 30)
        self.fc4 = nn.Linear(30, 4)   
        
        self.softmax = nn.Softmax(dim=0)
        
        
    def forward(self,head,body):
                
        head = torch.tensor(head).float()
        body = torch.tensor(body).float()
                
        key_head = self.Q_head * head
        query_head = self.K_head * head
        value_head = self.V_head * head
        
        key_body = self.Q_body  * body
        query_body = self.K_body  * body
        value_body = self.V_body * body
    
        head_attention  = self.attention_head(key_head,query_head,value_head) 
        body_attention  = self.attention_body(key_body,query_body,value_body)
        
        
        merge = torch.cat((head_attention,body_attention),axis=1).reshape(self.batch_size,60000)
                
        
        output = self.fc1(merge)
        output = self.fc2(output)
                
        output = self.fc3(output)
        output = self.fc4(output)

        output = self.softmax(output)
        
        return output
       
