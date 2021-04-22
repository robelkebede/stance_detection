import torch 

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
