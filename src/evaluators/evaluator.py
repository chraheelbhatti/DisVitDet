import torch
from torch.utils.data import DataLoader
from src.models.disvitdet import DisViTDet
from src.datasets.social_media_dataset import SocialMediaDataset
from src.utils.metrics import compute_metrics

class Evaluator:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model = DisViTDet(config['model'])
        self.dataset = SocialMediaDataset(config['data']['test_data_path'], config)
        self.dataloader = DataLoader(self.dataset, batch_size=config['training']['batch_size'], shuffle=False)

    def run(self):
        self.model.eval()
        with torch.no_grad():
            for memes, texts, emojis in self.dataloader:
                outputs = self.model(memes, texts, emojis)
                metrics = compute_metrics(outputs, labels)
                self.logger.info(f'Metrics: {metrics}')
