import torch

# Save Callbacks
def Save_Callbacks(checkpoint_path,optimizer,model,epoch):
    
    checkpoint={"Epoch":epoch,
                "Model_states":model.state_dict(),
                "Optimizer_states":optimizer.state_dict()}
    
    torch.save(checkpoint,f=checkpoint_path)
    
    print("Checkpoints are saved...")
    

# Load Callbacks
def Load_Callbakcs(model,optimizer,checkpoint):
    
    model.load_state_dict(checkpoint["Model_states"])
    optimizer.load_state_dict(checkpoint["Optimizer_states"])
    
    start_epoch=checkpoint["Epoch"]
    
    return start_epoch



    
    
    
    
    