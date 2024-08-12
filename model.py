from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("model/", do_lower_case=True)
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

model = BertForSequenceClassification.from_pretrained("model/")# Defining model
tokenizer = BertTokenizer.from_pretrained("model/")
import os


import numpy as np
import os
import random
import torch

Doctor = [
    "Doctor Shah",
    "Doctor Firoz",
    "Doctor Gandhi",
    "Doctor Kuber",
    "Doctor Singhaniya",
]
Quotes = [
    "Go with the Flow, or breeze  will continue and let you Blow",
    "Try.. if not Try harder.. Still not working, take your time out and time will handle it ",
    "Life is Bliss, and so are you!! ",
    "Life is a beautiful thing, perhaps better!",
    "Bird and sunny sky after heavy rain... Mind is refreshed!",
]
d = random.randint(0, 4)
q = random.randint(0, 4)


def Sentiment(sent):

    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=64,#max length
        pad_to_max_length=True,# padding
        return_attention_mask=True,# Attention Mask [Padding -> 0 / Token -> 1]
        return_tensors="pt",
    )

    input_id = encoded_dict["input_ids"] #input 

    attention_mask = encoded_dict["attention_mask"] # input

    # Torch format conversion
    input_id = torch.LongTensor(input_id)
    attention_mask = torch.LongTensor(attention_mask)

    with torch.no_grad(): # Eval mode
        outputs = model(input_id, token_type_ids=None, attention_mask=attention_mask)

    logits = outputs[0] # Logit -> Answer
    index = logits.argmax() # Probablity round off
    if index == 1:
        r = "Positive !! Great You are not in depression!!"
    else:
        r = f"Negative, You are in depression, \n  \n  \n  {Quotes[q]} \n \n \n  consider consulting {Doctor[d]}"
    return r
