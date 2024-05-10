import torch
from torch import nn
from transformers import AutoModel


class BERTClassifier(nn.Module):
    def __init__(self, num_labels=1):        
        super().__init__()        
        self.distilBERT = AutoModel.from_pretrained("distilbert-base-uncased")

        for param in self.distilBERT.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(self.distilBERT.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.distilBERT(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        logits = self.classifier(hidden_state)
        return logits


class BERTClassifierUnfrozen(nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.distilBERT = AutoModel.from_pretrained("distilbert-base-uncased")

        for param in self.distilBERT.parameters():
            param.requires_grad = True

        self.classifier = nn.Linear(self.distilBERT.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.distilBERT(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        logits = self.classifier(hidden_state)
        return logits
