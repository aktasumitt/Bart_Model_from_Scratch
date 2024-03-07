import torch,tqdm

def Test(Model,Test_dataloader,loss_fn,devices):
    loss_total=0.0
    total_size=0.0
    total_correct=0.0
    
    test_bar=tqdm.tqdm(Test_dataloader,"Test Progress")
    for batch,(input_encoder,out_decoder) in enumerate(Test_dataloader):
        
        input_encoder=input_encoder.to(devices)
        out_decoder=out_decoder.to(devices)
        
        out=Model(input_encoder)
        loss=loss_fn(out.reshape(-1,out.size(2)),out_decoder.reshape(-1))
        
        _,pred=torch.max(out,-1)
        total_correct+=(pred==out_decoder).sum().item()
        total_size+=(out.size(0)*out.size(1))
        loss_total+=loss.item()
        test_bar.update()
    
    test_bar.set_postfix({"Loss_Test":loss_total/(batch+1),
                          "Acc_Test":100*(total_correct/total_size)})
    test_bar.close()
    
        
        
    