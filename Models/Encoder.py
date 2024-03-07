import torch
import torch.nn as nn

class Encoder_Model(nn.Module):
    
    def __init__(self,d_model,num_heads,devices,batch_size):
        super(Encoder_Model,self).__init__()
        
        self.batch_size=batch_size
        self.devices=devices                # device "cuda or cpu"
        self.d_model=d_model                # d_model for model layers total unit size => (512 in article) 
        self.num_heads=num_heads            # num_heads is number of head (repetitions) for query, value and key
        self.d_k=int(self.d_model/num_heads)    # d_k is for scaled dot product and (query, key, value) unit size
                
        # Multihead Attention
        self.query=torch.nn.Linear(in_features=self.d_model,out_features=self.d_model) 
        self.key=torch.nn.Linear(in_features=self.d_model,out_features=self.d_model)     
        self.value=torch.nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.concat_scaled_dot_product=torch.nn.Linear(d_model,d_model) 

        # Feed Forward
        self.feed_forward1=torch.nn.Linear(self.d_model,2048)
        self.feed_forward2=torch.nn.Linear(2048,self.d_model)

    
    # Multihead Attention
    def Multihead_Attention(self,data):        
            
        query=self.query(data).reshape(self.batch_size,-1,self.num_heads,self.d_k).permute(0,2,1,3) 
        key=self.key(data).reshape(self.batch_size,-1,self.num_heads,self.d_k).permute(0,2,1,3)    
        value=self.value(data).reshape(self.batch_size,-1,self.num_heads,self.d_k).permute(0,2,1,3) 
            
        dot_product=torch.matmul(query,torch.transpose(key,-2,-1))/((self.d_model/self.num_heads)**1/2) 
        scaled_dot=torch.nn.functional.softmax(dot_product,dim=-1)   
        
        scaled_dot=torch.matmul(scaled_dot,value) 
        
        scaled_dot=scaled_dot.permute(0,2,1,3)
        
        concat_scaled_dot_product=scaled_dot.reshape_as(data)  # concatinate
        
        concat_scaled_dot_product=self.concat_scaled_dot_product(concat_scaled_dot_product) 
        
        return concat_scaled_dot_product
    
    # Feed FOrward
    def Feed_Forward(self,data):
    
        data=self.feed_forward1(data)
        data=torch.nn.functional.gelu(data)
        data=self.feed_forward2(data)
        
        return data
    
    def forward(self,data):
            
        mhe_data=self.Multihead_Attention(data=data)
        norm_data=nn.functional.layer_norm((mhe_data+data),normalized_shape=mhe_data.shape) # layer Norm and residual
            
        feed_forward=self.Feed_Forward(data=norm_data)
        data=nn.functional.layer_norm((norm_data+feed_forward),normalized_shape=feed_forward.shape) # layer norm and residual

        return data
        