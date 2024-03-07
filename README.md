# BART Model From Scratch for Translate 

## Introduction:
In this project, I aimed to train a Custom Bart model for translate text from Turkish to English.

## Tensorboard:
TensorBoard, along with saving training or prediction images, allows you to save them in TensorBoard and examine the changes graphically during the training phase by recording scalar values such as loss and accuracy. It's a very useful and practical tool.

## Dataset:
- In this project, I used a dataset is containing aproximately 80k question-answer and context text for the model.
-  However, I haven't trained it yet because it requires a very high amount of time and it have huge size so I cannot share my training data here,

## Model:
- The model is based on the transformer and its model architecture is not significantly different from that of the transformer.
- Since I have already examined the transformer in detail in my previous repository, I will only touch upon the different aspects of the BART model.
- The BART model is like a combination of the BERT model and the GPT model. BERT utilizes only a Bidirectional Encoder, while GPT uses only a generative pre-trained decoder. BART employs both.
- The BART model differs from the transformer by replacing the ReLU activation function with the GELU function.
- It has two training phases: pre-training and fine-tuning.
-  In the pre-training phase, we feed randomly corrupted text to the model and ask it to generate uncorrupted text. This helps the model to extract the features of the language.
-  In the fine-tuning phase, the features learned in the pre-training phase are very helpful in generating real data. It has been observed that this stage significantly changes the score value.
-  In the fine-tuning phase, datasets specific to the task the model learned during pre-training are provided.
-  For tasks like question answering, context and question are combined with a special '[SEP]' token and fed to the model, expecting it to learn the output answer value.

#### Augment Dataset:
- BART has achieved significantly better results by manipulating the dataset through various techniques.
- With the masking method, it randomly replaces 15% of the tokens in the text with a '[MASK]' token with an 80% probability. Additionally, there's a 10% chance that any token will be randomly replaced with another token, and another 10% chance that no action will be taken.
- Through the infilling method, random portions of the text are selected based on a Poisson distribution (lambda = 3), and these portions are replaced with '[MASK]' tokens.
- The deletion method involves randomly removing tokens from 15% of the text.
- Although sentence permutation and sentence shuffling methods exist, they haven't yielded as successful results as the aforementioned ones, hence I haven't included them in the code.
- These data augmentation techniques have led to improved results.

#### For More details:
- Model: https://arxiv.org/abs/1910.13461
- Dataset : https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset

## Train:
- With the resources at my disposal, I haven't been able to train the model further because it requires a lot of time and resources. 
- For training, I used Kaggle Stanfor Question-Answerring dataset.
- I employed the Adam optimizer and categoricel cross-entropy loss for fine tuning and negative-log-likelihood loss for pre training

