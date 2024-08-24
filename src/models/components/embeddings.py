import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import vit_b_16

class TextEmbedding(nn.Module):
    def __init__(self, config):
        super(TextEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, text):
        outputs = self.bert(**text)
        return outputs.last_hidden_state

class MemeEmbedding(nn.Module):
    def __init__(self, config):
        super(MemeEmbedding, self).__init__()
        self.vit = vit_b_16(pretrained=True)

    def forward(self, meme):
        patches = self.vit(meme)
        return patches

class EmojiEmbedding(nn.Module):
    def __init__(self, config):
        super(EmojiEmbedding, self).__init__()
        self.emoji_embedding = nn.Embedding(config['emoji_vocab_size'], config['emoji_embedding_dim'])

    def forward(self, emojis):
        embeddings = self.emoji_embedding(emojis)
        return embeddings
