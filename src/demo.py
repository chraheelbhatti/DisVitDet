from src.models.disvitdet import DisViTDet
import torch

def demo():
    model = DisViTDet(config='experiments/config_disvitdet.json')
    meme = torch.randn((1, 3, 224, 224))  # Dummy input for meme image
    text = {'input_ids': torch.randint(0, 30522, (1, 20)), 'attention_mask': torch.ones((1, 20))}  # Dummy input for text
    emojis = torch.randint(0, 1000, (1, 5))  # Dummy input for emojis
    output = model(meme, text, emojis)
    print(f'Prediction: {output}')

if __name__ == '__main__':
    demo()
