import nltk
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.transforms import VocabTransform
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
import pandas as pd


# Loading Dataa
class Loading_Dataset():

    def __init__(self, Json_dataset_Path, max_input_data_len, max_output_data_len):

        self.input_data_len = max_input_data_len
        self.output_data_len = max_output_data_len

        # create pandas dataset
        self.json_dataset = pd.read_json(Json_dataset_Path)

    def Loading(self):
        # question answers samples
        sample_list = []
        for i in range(len(self.json_dataset["data"])):
            sample_list.append(self.json_dataset["data"][i]["paragraphs"])

        # sample seperated lists
        # input is "context + "[seperate_token]" + question" , i know seperate token is anormal but i will change in preprocess step
        input_list = []
        output_list = []              # output is answers
        answer_start_list = []        # word token of answer starting

        for pharagraphs in sample_list:
            for samples in pharagraphs:

                context = samples["context"]  # Context

                for item in samples["qas"]:

                    question = item["question"]  # question

                    for i in item["answers"]:

                        answer = i["text"]  # answer
                        answer_start = i["answer_start"]

                        input_list.append(
                            context+" "+"seppp_token"+" "+question)
                        output_list.append(answer)
                        answer_start_list.append(answer_start)

        return input_list, output_list, answer_start_list

    def Tresholder(self, input_list, output_list):

        # Delete input samples as the treshold
        for step, sample in enumerate(input_list):
            if len(sample.split(" ")) > self.input_data_len:
                del input_list[step]
                del output_list[step]

        for step, sample in enumerate(output_list):
            if len(sample.split(" ")) > self.output_data_len:
                del input_list[step]
                del output_list[step]

            return input_list, output_list

    def __call__(self):

        input_list, output_list, answer_start_list = self.Loading()
        input_list, output_list = self.Tresholder(input_list, output_list)

        return input_list, output_list, answer_start_list


# Preproccess text datas
class Preproccess():

    """
    We can tokenize and pad text datas with this class.
    mode = "deletion" or "masking" or "infill"
    """

    def __init__(self, max_len: int, language: str, mask_symbol:str=None, seperate_symbol:str=None, start_symbol: str = None, stop_symbol: str = None, pad_symbol: str = None, mode_aug: str = None):

        self.start_symbol = start_symbol
        self.stop_symbol = stop_symbol
        self.pad_symbol = pad_symbol
        self.seperate_symbol=seperate_symbol
        self.mask_symbol=mask_symbol
        self.max_len = max_len
        self.language = language
        self.mode = mode_aug
        
        
        
    # Tokenize Texts
    def Tokenize(self, Text):

        Tokenized = []

        for i in Text:
            
            timestep_list = nltk.tokenize.word_tokenize(
                i.lower(), language=self.language)
            
            
            if self.seperate_symbol!=None:
                for i, sample in enumerate(timestep_list):
                    if sample == "seppp_token":
                        timestep_list[i] = self.seperate_symbol # change SEP token

            if self.start_symbol != None:
                timestep_list.insert(0, self.start_symbol)
            if self.stop_symbol != None:
                timestep_list.append(self.stop_symbol)

            Tokenized.append(timestep_list)

        return Tokenized

    # Masking Data
    def Masking(self, data):

        # %80 change to "MASK" , %10 change another word , %10 do nothing
        prob_num = torch.randint(1, 100, (1,)).item()

        for batch in data:

            # change data (%15 of length of data) times
            change_num = int(len(batch)*15/100)

            for _ in range(change_num):
                
                # Random index number
                random_num = torch.randint(1, len(batch), (1,)).item() 
                
                # change to mask
                if prob_num <= 80:
                    batch[random_num] = self.mask_symbol

                # change to another word
                elif prob_num < 90 and prob_num > 80:

                    batch[random_num] = batch[torch.randint(
                        1, len(batch), (1,)).item()]

                # do nothing
                else:
                    pass

        return data

    # Token deletion
    def Token_Deletion(self, data):
        out_data = []

        for batch in data:
            
            # %80 delete random value
            prob_num = torch.randint(1, 100, (1,)).item()
            
            # delete value (%15 of length of data) times
            change_num = int(len(batch)*15/100)

            for _ in range(change_num):
                
                # Random index number for data
                random_num = torch.randint(1, len(batch), (1,)).item()
                
                # delete to random value
                if prob_num <= 80:
                    out = batch[:random_num]+batch[random_num+1:]

            out_data.append(out)

        return out_data

    
    # Text infiliing with random poisson distrubution
    def Text_infilling(self, data):

        out_data = []

        # Poisson distrubution with lamda=3
        poisson = torch.distributions.Poisson(3).sample_n(len(data))

        for i, batch in enumerate(data):

            # Get Poisson number for this index
            poisson_num = int(poisson[i].item())

            # random num
            random_num = torch.randint(1, len(batch), (1,)).item()

            # if poisson distrubution = 0 ,add mask
            if poisson_num == 0:
                batch[random_num] = self.mask_symbol
                out_data.append(batch)

            # choose random poison_noise size of word phrase and change with mask
            else:
                batch[random_num] = self.mask_symbol
                batch = batch[:random_num+1]+batch[random_num+poisson_num:]
                out_data.append(batch)

        return out_data

    # Padding if u want

    def Padding_Text(self, Tokenized_data):

        padded_list = []
        for i in Tokenized_data:
            padded_list.append(list(nltk.pad_sequence(sequence=i, n=(self.max_len-len(i)+1), pad_right=True, pad_left=False, right_pad_symbol=self.pad_symbol)))
        return padded_list

    # call class

    def __call__(self, text):

        print(" Texts are Preprocessing...")

        # Tokenize
        data_prep = self.Tokenize(Text=text)

        # Augmentation
        if self.mode == "masking":
            data_prep = self.Masking(data_prep)
        if self.mode == "deletion":
            data_prep = self.Token_Deletion(data_prep)
        if self.mode == "infill":
            data_prep = self.Text_infilling(data_prep)

        # Padding
        if self.pad_symbol != None:
            data_prep = self.Padding_Text(data_prep)

        return data_prep


class Create_Dataset(Dataset):
    def __init__(self, tokenized_data_in, tokenized_data_out,mask_symbol:str=None, seperate_symbol:str=None, start_symbol: str = None, stop_symbol: str = None, pad_symbol: str = None):
        super().__init__()
        self.tokenized_data_in = tokenized_data_in
        self.tokenized_data_out = tokenized_data_out
        self.start_symbol = start_symbol
        self.stop_symbol = stop_symbol
        self.pad_symbol = pad_symbol
        self.seperate_symbol=seperate_symbol
        self.mask_symbol=mask_symbol

        self.vocab_in = build_vocab_from_iterator(
            self.tokenized_data_in, special_first=True, specials=[self.pad_symbol, 
                                                                  self.start_symbol,
                                                                  self.stop_symbol,
                                                                  self.mask_symbol,
                                                                  self.seperate_symbol])
        self.vocab_out = build_vocab_from_iterator(
            self.tokenized_data_out, special_first=True, specials=[self.pad_symbol, 
                                                                  self.start_symbol,
                                                                  self.stop_symbol,
                                                                  self.mask_symbol,
                                                                  self.seperate_symbol])

        self.data_in = VocabTransform(self.vocab_in)(self.tokenized_data_in)
        self.data_out = VocabTransform(self.vocab_out)(self.tokenized_data_out)

    def __len__(self):
        return len(self.tokenized_data_in)

    def word2idx_out(self):
        # creating word2idx dict for output
        return self.vocab_out.get_stoi()

    def word2idx_input(self):
        # creating word2idx dict for input
        return self.vocab_in.get_stoi()

    def __getitem__(self, index):

        return (torch.tensor(self.data_in[index]), torch.tensor(self.data_out[index]))



# Random split
def random_split_fn(dataset, valid_range):
    valid_size = int(len(dataset)*valid_range)

    Train, Valid, Test = random_split(
        dataset, [len(dataset)-(valid_size*2), valid_size, valid_size])
    return Train, Valid, Test


# Dataloader
def Dataloader_fn(train, valid, test, batch_size):

    train = DataLoader(dataset=train, batch_size=batch_size,
                       shuffle=True, drop_last=True)
    valid = DataLoader(dataset=valid, batch_size=batch_size,
                       shuffle=False, drop_last=True)
    test = DataLoader(dataset=test, batch_size=batch_size,
                      shuffle=False, drop_last=True)

    return train, valid, test
