import torch
import wandb
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel


class SamplesVisualisationLogger(pl.Callback):
    """About Explanation How to Use PytorchLightning

    Callbacks: https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html


    """
    def __init__(self, datamodule: DataModule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        """
        on_validation_end:

        https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#on-validation-end

        pl.Trainer.logger:
        https://lightning.ai/docs/pytorch/stable/common/trainer.html#lightning.pytorch.trainer.Trainer.params.logger

        WandbLogger.experiment:
        https://lightning.ai/docs/pytorch/latest/extensions/generated/lightning.pytorch.loggers.WandbLogger.html#lightning.pytorch.loggers.WandbLogger.params.experiment

        WandbLogger.experiment.log:
        https://lightning.ai/docs/pytorch/latest/extensions/generated/lightning.pytorch.loggers.WandbLogger.html#lightning.pytorch.loggers.WandbLogger.params.experiment

        wandb.Table:
        https://docs.wandb.ai/guides/track/log/log-tables

        trainer.global_step:
        https://lightning.ai/docs/pytorch/stable/common/trainer.html#global-step
        """
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", # on wandb or on local? I'm not sure whether it is.
        filename="best-checkpoint.ckpt",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    # https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.loggers.wandb.html#weights-and-biases-logger
    wandb_logger = WandbLogger(project="MLOps Basics", entity="raviraja")
    trainer = pl.Trainer(
        max_epochs=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(cola_data), early_stopping_callback],
        log_every_n_steps=10,
        deterministic=True,
        # limit_train_batches=0.25,
        # limit_val_batches=0.25
    )
    # I think I have to implement seed_everything
    #   https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#fit
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()
