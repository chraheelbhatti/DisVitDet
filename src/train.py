from src.models.disvitdet import DisViTDet
from src.datasets.social_media_dataset import SocialMediaDataset
from src.utils.metrics import compute_metrics
from torch.utils.data import DataLoader
import torch

def train(config, logger):
    model = DisViTDet(config['model'])
    dataset = SocialMediaDataset(config['data']['train_data_path'], config)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config['training']['num_epochs']):
        for memes, texts, emojis, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(memes, texts, emojis)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        logger.info(f'Epoch {epoch + 1}, Loss: {loss.item()}')
