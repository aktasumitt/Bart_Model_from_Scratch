from Models import Decoder,Encoder,Bart_Model,Embedding
import dataset
import Pre_Train
import config
import callbakcs
import torch
import test
import prediction
import Fine_Tune
import warnings
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

# Tensorboard
Tensorboard_Writer = SummaryWriter()

# Devices to use cuda
devices = ("cuda" if torch.cuda.is_available() else "cpu")


# Loading Datasets from scratch
input_list,output_list,answer_start_list=dataset.Loading_Dataset(Json_dataset_Path="Dataset\\train-v1.1.json",
                                                                 max_input_data_len=config.TRESHOLD_LEN_INPUT,
                                                                 max_output_data_len=config.TRESHOLD_LEN_OUTPUT)()

# Preprocess Texts for Pre-Training:
Input_data = dataset.Preproccess(max_len=config.TRESHOLD_LEN_INPUT,
                                language=config.LANGUAGE,
                                start_symbol=config.START_SYMBOL,
                                stop_symbol=config.STOP_SYMBOL,
                                pad_symbol=config.PAD_SYMBOL,
                                seperate_symbol=config.SEPERATE_SYMBOL,
                                mask_symbol=config.MASK_SYMBOL,
                                mode_aug="masking")(text=input_list)

# No Augmentation for output pretrain
Output_data_Pretrain=dataset.Preproccess(max_len=config.TRESHOLD_LEN_INPUT,
                                        language=config.LANGUAGE,
                                        stop_symbol=config.STOP_SYMBOL,
                                        pad_symbol=config.PAD_SYMBOL,
                                        seperate_symbol=config.SEPERATE_SYMBOL,
                                        mask_symbol=config.MASK_SYMBOL)(text=input_list)


# OUTPUT for  Fine-tuning
Out_data = dataset.Preproccess(max_len=config.TRESHOLD_LEN_OUTPUT,
                                language=config.LANGUAGE,
                                stop_symbol=config.STOP_SYMBOL,
                                pad_symbol=config.PAD_SYMBOL)(text=output_list)

# Create Dataset
Train_dataset_Finetune=dataset.Create_Dataset(tokenized_data_in=Input_data,
                                                tokenized_data_out=Out_data,
                                                start_symbol=config.START_SYMBOL,
                                                stop_symbol=config.STOP_SYMBOL,
                                                pad_symbol=config.PAD_SYMBOL,
                                                seperate_symbol=config.SEPERATE_SYMBOL,
                                                mask_symbol=config.MASK_SYMBOL)

Train_Dataset_Pretrain=dataset.Create_Dataset(tokenized_data_in=Input_data,
                                                tokenized_data_out=Output_data_Pretrain,
                                                start_symbol=config.START_SYMBOL,
                                                stop_symbol=config.STOP_SYMBOL,
                                                pad_symbol=config.PAD_SYMBOL,
                                                seperate_symbol=config.SEPERATE_SYMBOL,
                                                mask_symbol=config.MASK_SYMBOL)


# WORD to IDX DICTIONARIES
W2IDX_IN = Train_dataset_Finetune.word2idx_input()
W2IDX_OUT = Train_Dataset_Pretrain.word2idx_out()


# Random Split for Pretraining and Finetuning
Train_dataset_pre, Valid_dataset_pre, Test_dataset_pre = dataset.random_split_fn(dataset=Train_Dataset_Pretrain,
                                                                     valid_range=config.VALID_RANGE)
Train_dataset_fine, Valid_dataset_fine, Test_dataset_fine = dataset.random_split_fn(dataset=Train_dataset_Finetune,
                                                                     valid_range=config.VALID_RANGE)

# Create Dataloaders:
Train_dataloader_pre, Valid_Dataloader_pre, Test_Dataloader_pre = dataset.Dataloader_fn(train=Train_dataset_pre,
                                                                                valid=Valid_dataset_pre,
                                                                                test=Test_dataset_pre,
                                                                                batch_size=config.BATCH_SIZE)

Train_dataloader_fine, Valid_Dataloader_fine, Test_Dataloader_fine = dataset.Dataloader_fn(train=Train_dataset_fine,
                                                                            valid=Valid_dataset_fine,
                                                                            test=Test_dataset_fine,
                                                                            batch_size=config.BATCH_SIZE)
# Create Model:
Model = Bart_Model.Bart_Model(d_model=config.D_MODEL,
                              Encoder_Model=Encoder.Encoder_Model,
                              Decoder_Model=Decoder.Decoder_Model,
                              Embeddig_Model=Embedding.Embedding_Model,
                              vocab_size_encoder=len(W2IDX_IN)+2,
                              vocab_size_decoder=len(W2IDX_OUT)+2,
                              num_heads=config.NUM_HEADS,
                              pad_idx=config.PAD_IDX,
                              max_seq_len_input=config.TRESHOLD_LEN_INPUT,
                              max_seq_len_output=config.TRESHOLD_LEN_OUTPUT,
                              devices=devices,
                              batch_size=config.BATCH_SIZE,
                              masking_value=config.MASKING_VALUE,
                              Nx=config.NX,
                              stop_token=W2IDX_IN[config.STOP_SYMBOL])
Model.to(devices)
Model.train()


# Create Optimizer and Loss_fn
optimizers = torch.optim.Adam(params=Model.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS, eps=config.EPSILON)
loss_fn_fine_tune = torch.nn.CrossEntropyLoss(reduction="sum")
loss_fn_pre_train= torch.nn.NLLLoss(reduction="sum")


# Load Callbacks for pre train and starting fine tuning
if config.LOAD_CALLBACKS_PRETRAIN == True:
    print("Callbacks are Loading...")
    checkpoint = torch.load(f=config.PRETRAIN_CALLBACKS_PATH)
    start_epoch = callbakcs.Load_Callbakcs(
        model=Model, optimizer=optimizers, checkpoint=checkpoint)
else:
    start_epoch = 0


# Pre-Train or Fine Tuning
if config.PRE_TRAIN == True:
    Pre_Train.Pre_train(Train_Dataloader=Train_dataloader_pre,
                        Valid_Dataloader=Valid_Dataloader_pre,
                        optimizer=optimizers,
                        loss_fn=loss_fn_pre_train,
                        model=Model,
                        epochs=config.EPOCHS_PRETRAIN,
                        start_epochs=start_epoch,
                        devices=devices,
                        save_callbacks=callbakcs.Save_Callbacks,
                        checkpoint_path=config.PRETRAIN_CALLBACKS_PATH,
                        Tensorboard=Tensorboard_Writer)


# Load Callbacks fine tune
if config.LOAD_CALLBACKS_FINETUNE== True:
    print("Callbacks are Loading...")
    checkpoint = torch.load(f=config.FINETUNE_CALLBACKS_PATH)
    start_epoch = callbakcs.Load_Callbakcs(
    model=Model, optimizer=optimizers, checkpoint=checkpoint)
else:
    start_epoch = 0


# Fine Tuning 
# we need load pretrain_callbacks to start training fine tune
if config.FINE_TUNE:
    Fine_Tune.Fine_Tuning(Train_Dataloader=Train_dataloader_fine,
                          Valid_Dataloader=Valid_Dataloader_fine,
                          optimizer=optimizers,
                          loss_fn=loss_fn_fine_tune,
                          model=Model,
                          epochs=config.EPOCH_FINETUNE,
                          devices=devices,
                          vocab_size_out=len(W2IDX_OUT),
                          save_callbacks=callbakcs.Save_Callbacks,
                          checkpoint_path=config.FINETUNE_CALLBACKS_PATH,
                          Tensorboard=Tensorboard_Writer)

# TEST
if config.TEST == True:
    
    test.Test(Model=Model, Test_dataloader=Test_Dataloader_fine,
              loss_fn=loss_fn_fine_tune, devices=devices)


# PREDICTION
if config.PREDICTION == True:
    translate = prediction.prediction(sentence=config.PREDICTION_SENTENCE,
                                      Model=Model,
                                      word2idx_in=W2IDX_IN,
                                      word2idx_out=W2IDX_OUT,
                                      padding_len=config.TRESHOLD_LEN_INPUT,
                                      pad_idx=config.PAD_IDX,
                                      devices=devices)
