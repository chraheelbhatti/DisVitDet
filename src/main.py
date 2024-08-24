import argparse
from src.trainers.trainer import Trainer
from src.evaluators.evaluator import Evaluator
from src.utils.logger import setup_logger
from src.config import load_config

def main():
    parser = argparse.ArgumentParser(description="DisViTDet: Multi-Modal Vision Transformer for Disinformation Detection")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Run mode: train or test')
    parser.add_argument('--config', type=str, default='experiments/config_disvitdet.json', help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger()

    if args.mode == 'train':
        trainer = Trainer(config, logger)
        trainer.run()
    elif args.mode == 'test':
        evaluator = Evaluator(config, logger)
        evaluator.run()

if __name__ == '__main__':
    main()
