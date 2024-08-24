from src.models.disvitdet import DisViTDet
from src.datasets.social_media_dataset import SocialMediaDataset
from src.utils.metrics import compute_metrics
from torch.utils.data import DataLoader
import torch

def test(config, logger):
    model = DisViTDet(config['model'])
    dataset = SocialMediaDataset(config['data']['test_data_path'], config)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)

    model.eval()
    with torch.no_grad():
        for memes, texts, emojis, labels in dataloader:
            outputs = model(memes, texts, emojis)
            metrics = compute_metrics(outputs, labels)
            logger.info(f'Metrics: {metrics}')
