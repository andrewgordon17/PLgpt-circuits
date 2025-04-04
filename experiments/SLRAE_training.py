import argparse
from pathlib import Path

import torch

from config import TrainingConfig
from config.sae.training import SAETrainingConfig, options
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput
from training.sae import SAETrainer
from training.sae.concurrent import ConcurrentTrainer

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training config")
    parser.add_argument("--load_from", type=str, help="GPT model weights to load")
    parser.add_argument("--name", type=str, help="Model name for checkpoints")
    parser.add_argument("--rank_bound", type=int, help="Low Rank Bound")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = "standardLR.shakespeare_64x4"
    config = options[config_name]

    ranks = [0,1,2,3,4,5,6,7,8,9] #MODIFY THIS and ONLY THIS

    # Initialize trainer
    #trainer = ConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
    #trainer.train()

    for rank_bound in [30,60, 64]:
        # Initialize trainer
        config.sae_config.rank_bound = rank_bound
        config.name = f"SLRAE-rank{rank_bound}"
        config.out_dir.mkdir(parents=True, exist_ok=True)
        ############# Set up the model #############
        trainer = ConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
        trainer.train()