import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(config['hidden_size'], config['attention_size'])
        self.key = nn.Linear(config['hidden_size'], config['attention_size'])
        self.value = nn.Linear(config['hidden_size'], config['attention_size'])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, meme_features, text_features, emoji_features):
        query = self.query(meme_features)
        key = self.key(text_features)
        value = self.value(emoji_features)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_weights = self.softmax(attention_scores)
        attended_features = torch.matmul(attention_weights, value)
        return attended_features
