import torch.nn as nn
import torch



class Decoder_Model(nn.Module):
    """
    Decoder_Model input to forward is tuple that have encoder_output and decoder_input like that (encoder_output,decoder_input)
    """
   
    def __init__(self,devices,d_model:int,num_heads:int,batch_size:int,masking_value:int=-1e8):
        super(Decoder_Model,self).__init__()
        
        self.batch_size=batch_size          # Batch_size of model
        self.masking_value=masking_value    # masking value that is too big value on inf
        self.devices=devices                # device "cuda or cpu"
        self.d_model=d_model                # d_model for model layers total unit size => (512 in article )
        self.num_heads=num_heads                # num_heads is number of head (repetitions) for query, value and key
        self.d_k=int(self.d_model/num_heads)    # d_k is for scaled dot product and (query, key, value) unit size
        
        
        # Masked Multi Head Attention
        self.query_m=torch.nn.Linear(in_features=self.d_model,out_features=d_model) 
        self.key_m=torch.nn.Linear(in_features=self.d_model,out_features=self.d_model)     
        self.value_m=torch.nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.concat_scaled_dot_product_m=torch.nn.Linear(d_model,d_model) 

        # Multihead Attention
        self.query=torch.nn.Linear(in_features=self.d_model,out_features=self.d_model) 
        self.key=torch.nn.Linear(in_features=self.d_model,out_features=self.d_model)     
        self.value=torch.nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.concat_scaled_dot_product=torch.nn.Linear(d_model,d_model) 

        # Feed Forward
        self.feed_forward1=torch.nn.Linear(self.d_model,2048)
        self.feed_forward2=torch.nn.Linear(2048,self.d_model)
        
   
    # Masked Multi Head Attention
    def Masked_Multihead_Attention(self,data):
   
        query=self.query_m(data).reshape(self.batch_size,-1,self.num_heads,self.d_k).permute(0,2,1,3)    #(batch,10,64) 10=max_len
        key=self.key_m(data).reshape(self.batch_size,-1,self.num_heads,self.d_k).permute(0,2,1,3)   
        value=self.value_m(data).reshape(self.batch_size,-1,self.num_heads,self.d_k).permute(0,2,1,3) 
            
        dot_product=torch.matmul(query,torch.transpose(key,-1,-2))/(self.d_k**1/2) #(batch,10,10) 
         
        # We Use masked with masking_value that is too big negative or positive
        mask=torch.triu(torch.ones_like(dot_product),diagonal=1).to(self.devices)
        mask_data=self.masking_value*mask   
        
        # Masked
        masked_product=mask_data+dot_product # (batch,10,10)
            
        scaled_dot=torch.nn.functional.softmax(masked_product,dim=-1)  #(batch,10,10)
        scaled_dot=torch.matmul(scaled_dot,value)    #(batch,10,64)
        
        scaled_dot=scaled_dot.permute(0,2,1,3)
        
        concat_scaled_dot_product=scaled_dot.reshape((self.batch_size,data.shape[1],-1)) 

        concat_scaled_dot_product=self.concat_scaled_dot_product_m(concat_scaled_dot_product) #(batch,10,512)

        return concat_scaled_dot_product
    
    
    # Multi Head Attention
    def Multihead_Attention(self,data,encoder_out):

        query=self.query(data).reshape(self.batch_size,-1,self.num_heads,self.d_k).permute(0,2,1,3)   
        key=self.key(encoder_out).reshape(self.batch_size,-1,self.num_heads,self.d_k).permute(0,2,1,3)   
        value=self.value(encoder_out).reshape(self.batch_size,-1,self.num_heads,self.d_k).permute(0,2,1,3) 
            
        dot_product=torch.matmul(query,torch.transpose(key,-1,-2))/((self.d_model/self.num_heads)**1/2) 
        scaled_dot=torch.nn.functional.softmax(dot_product,dim=-1)              
        scaled_dot=torch.matmul(scaled_dot,value)  
        
        scaled_dot=scaled_dot.permute(0,2,1,3)
        
        concat_scaled_dot_product=scaled_dot.reshape((self.batch_size,data.shape[1],-1))  
        
        concat_scaled_dot_product=self.concat_scaled_dot_product(concat_scaled_dot_product) 
        
        return concat_scaled_dot_product
    
    
    # Feed Forward
    def Feed_Forward(self,data):
    
        data=self.feed_forward1(data)
        data=torch.nn.functional.gelu(data)
        data=self.feed_forward2(data)
        
        return data
    
    
    def forward(self,data):
        encoder_out,data_dec=data
            
        mmhe_data=self.Masked_Multihead_Attention(data=data_dec)
        norm_mmhe=nn.functional.layer_norm((mmhe_data+data_dec),normalized_shape=mmhe_data.shape)

        mhe_data=self.Multihead_Attention(data=norm_mmhe,encoder_out=encoder_out)
        norm_mhe=nn.functional.layer_norm((mhe_data+norm_mmhe),normalized_shape=mhe_data.shape)
            
        feed_forward=self.Feed_Forward(data=norm_mhe)
        data_dec=nn.functional.layer_norm((norm_mhe+feed_forward),normalized_shape=feed_forward.shape)
        
        return data_dec


        