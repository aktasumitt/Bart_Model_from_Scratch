import torch
import tqdm


def Pre_train(Train_Dataloader,Valid_Dataloader,start_epochs,model,optimizer,loss_fn,devices,epochs,save_callbacks,checkpoint_path,Tensorboard):
    
    print(f"Training is starting from {start_epochs+1}.epoch")
    
    for epoch in range(start_epochs,epochs):
       
        progress_bar=tqdm.tqdm(range(len(Train_Dataloader)),"Train Progress",leave=True)
        
        correct_train=0.0
        total_train=0.0
        loss_train=0.0
        
        # Start Train       
        for batch,(input_encoder,labels_Decoder) in enumerate(Train_Dataloader):
            
            correct_train=0.0
            total_train=0.0
            loss_train=0.0

            # We have input encoder and labels decoder
            input_encoder=input_encoder.to(devices)
            labels_Decoder=labels_Decoder.to(devices)
            
            # Create Model Out
            out=model(input_encoder)
            
            # Forward and Backward
            optimizer.zero_grad()
            loss=loss_fn(out.reshape(out.size(0)*out.size(1),out.size(2)),input_encoder[:,1:].reshape(-1))  # we add stop token in output so we need stop token for loss in decoder targets
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # Calculate loss and accuracy
            _,pred=torch.max(out,-1)
            loss_train+=loss.item()
            correct_train+=(pred==labels_Decoder).sum().item()
            total_train+=(out.size(0)*out.size(1))    
            
            
            progress_bar.update(1) # Progress bar update          
            
            # Validation Step
            if batch%5==4:
                
                correct_val=0.0
                total_val=0.0
                loss_val=0.0
                
                with torch.no_grad():
                    
                    for batch_val,(input_encoder_val,labels_Decoder_val) in enumerate(Valid_Dataloader):
                        
                        labels_Decoder_val=labels_Decoder_val.to(devices)
                        input_encoder_val=input_encoder_val.to(devices)            
                        
                        out_val=model(input_encoder_val)
                        loss_v=loss_fn(out_val.reshape(out_val.size(0)*out_val.size(1),out_val.size(2)),labels_Decoder_val.reshape(-1))  # we didnt add stop token in out so we remote stop token in labels_decoder in loss

                        _,pred_val=torch.max(out_val,-1)
                        loss_val+=loss_v.item()
                        correct_val+=(pred_val==labels_Decoder_val).sum().item()
                        total_val+=(out_val.size(0)*out_val.size(1))               
                        
                # Progress bar update
                progress_bar.set_postfix({"Loss_val":(loss_val/(batch_val+1)),
                                          "Acc_Val":((100*correct_val/total_val)),
                                          "Loss_Train":(loss_train/(batch+1)),
                                          "Acc_Train":((100*correct_train/total_train))})
                # Tensorboard Update
                Tensorboard.add_scalars("Accuracy and Loss For Training",
                                        {"Loss_val":(loss_val/(batch_val+1)),
                                          "Acc_Val":((100*correct_val/total_val)),
                                          "Loss_Train":(loss_train/(batch+1)),
                                          "Acc_Train":((100*correct_train/total_train))},
                                        global_step=(epoch+1)*(batch+1))                                                           
                        
            
        
        progress_bar.close()            
        
        # Saving Checkpoint every epochs  
        save_callbacks(checkpoint_path,optimizer,model,epoch+1)
