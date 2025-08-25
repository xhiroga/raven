import logging
import os

import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
import torch

from data.data_module import DataModule
from finetune_learner import Learner


# static vars
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("lightning").propagate = False


@hydra.main(config_path="conf", config_name="config_test")
def main(cfg):
    print(f"{cfg=}")
    if cfg.fix_seed:
        seed_everything(42, workers=True)

    if "SLURM_JOB_ID" in os.environ:
        print("The SLURM job ID for this run is {}".format(os.environ["SLURM_JOB_ID"]))
        cfg.slurm_job_id = os.environ["SLURM_JOB_ID"]
    else:
        print("Running locally (no SLURM)")
        cfg.slurm_job_id = None

    cfg.gpus = torch.cuda.device_count()
    print("num gpus:", cfg.gpus)

    wandb_logger = None
    if cfg.log_wandb:
        wandb_logger = instantiate(cfg.logger)

    data_module = DataModule(cfg)
    learner = Learner(cfg)

    trainer = Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        strategy=DDPStrategy(find_unused_parameters=False) if cfg.gpus > 1 else "auto"
    )

    trainer.test(learner, datamodule=data_module)


if __name__ == "__main__":
    main()
