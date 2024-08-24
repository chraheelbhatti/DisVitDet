import torch
from torch.utils.data import DataLoader
from src.models.disvitdet import DisViTDet
from src.datasets.social_media_dataset import SocialMediaDataset
from src.utils.metrics import compute_metrics

class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model = DisViTDet(config['model'])
        self.train_dataset = SocialMediaDataset(config['data']['train_data_path'], config)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        self.criterion = torch.nn.CrossEntropyLoss()

    def run(self):
        self.model.train()
        for epoch in range(self.config['training']['num_epochs']):
            for memes, texts, emojis, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(memes, texts, emojis)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            self.logger.info(f'Epoch {epoch+1}, Loss: {loss.item()}')
