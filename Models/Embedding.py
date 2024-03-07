import torch
import torch.nn as nn

class Embedding_Model(nn.Module):
    
    def __init__(self,vocab_size,d_model,pad_idx,devices,max_seq_len):
        super(Embedding_Model,self).__init__()
        self.d_model=d_model
        self.max_seq_len=max_seq_len
        self.devices=devices
        
        self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model,padding_idx=pad_idx)
    
    def Positional_Encoding(self):
        
        position=torch.arange(0,self.max_seq_len).reshape((self.max_seq_len,1))
        even_i=torch.arange(0,self.d_model,2) # çift sıra
        odd_i=torch.arange(1,self.d_model,2) # tek sıra
        
        pow_even=torch.pow(10000,(2*even_i/self.d_model))   
        pow_odd=torch.pow(10000,(2*odd_i/self.d_model)) 
        
        PE_even=torch.sin(position/pow_even)
        PE_odd=torch.cos(position/pow_odd)
        
        PE_full=torch.stack([PE_even,PE_odd],dim=2)
        PE_full=torch.flatten(PE_full,start_dim=1,end_dim=2)
        PE_full=torch.unsqueeze(PE_full,0)

        PE_full=PE_full.to(device=self.devices)
        
        return PE_full
    
    def forward(self,data):
        
        embed=self.embedding(data)
        PE=self.Positional_Encoding()
        out=embed+PE[:,:embed.shape[1],:]
        
        return out
        