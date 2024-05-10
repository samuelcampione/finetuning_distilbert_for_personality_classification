import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
# ! pip install langdetect -y
from langdetect import detect


def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    url_pattern = r'https?://\S+|www\.\S+' # Remove all URLs
    text = re.sub(url_pattern, '', text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def drop_short_and_too_long_texts(X, y):
    X_sub = []
    y_sub = []
    for text, label in zip(X, y):
        text_sub = []
        for word in text.split(" "):
            if len(word) > 3 and len(word) < 25:
                text_sub.append(word)
        if len(text_sub) > 3:
            X_sub.append(" ".join(text_sub))
            y_sub.append(label)
    return X_sub, y_sub


def drop_non_english_texts(X, y):
    X_sub = []
    y_sub = []
    for i, (text, label) in enumerate(zip(X, y)):
        if detect(text) == 'en':
            X_sub.append(text)
            y_sub.append(label)
    return X_sub, y_sub


def batch_encode(tokenizer, texts, batch_size=200, max_length=128):
    input_ids = []
    attention_mask = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='max_length',
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])    
    
    return input_ids, attention_mask


class PersonalityDataset(Dataset):
    def __init__(self, inputs, attentions, targets):
        self.inputs = torch.tensor(inputs)
        self.attentions = torch.tensor(attentions)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.attentions[idx], self.targets[idx]

