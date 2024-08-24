import torch
import torch.nn as nn
from src.models.components.attention import CrossAttention
from src.models.components.transformer import TransformerEncoder
from src.models.components.embeddings import TextEmbedding, MemeEmbedding, EmojiEmbedding

class DisViTDet(nn.Module):
    def __init__(self, config):
        super(DisViTDet, self).__init__()
        self.text_embedding = TextEmbedding(config)
        self.meme_embedding = MemeEmbedding(config)
        self.emoji_embedding = EmojiEmbedding(config)
        self.cross_attention = CrossAttention(config)
        self.transformer = TransformerEncoder(config)
        self.classifier = nn.Linear(config['transformer_hidden_size'], config['num_classes'])

    def forward(self, meme, text, emojis):
        meme_features = self.meme_embedding(meme)
        text_features = self.text_embedding(text)
        emoji_features = self.emoji_embedding(emojis)
        cross_attended = self.cross_attention(meme_features, text_features, emoji_features)
        output = self.transformer(cross_attended)
        logits = self.classifier(output)
        return logits
