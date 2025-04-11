from typing import Any, Dict, Tuple, Literal

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from .simulators import BatchedLightSimulation
import wandb
from ..utils.plot import plot_waveform_predictions

class FlashRecoLitModule(LightningModule):
    """Example of a `LightningModule` for flash reconstruction.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        simulator: BatchedLightSimulation,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        regularizer: torch.nn.Module | None,
        scheduler_interval: Literal['epoch', 'step'],
        compile: bool,
        watch_grad: bool,
    ) -> None:
        """Initialize a `FlashRecoLitModule`.

        :param net: The model to train.
        :param simulator: The simulator to use for training.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param regularizer: The regularizer to use for training.
        :param scheduler_interval: The interval to use for the learning rate scheduler.
        :param compile: Whether to compile the model.

        The network MUST take in a tensor of shape (batch_size, num_channels, num_ticks),
        i.e., a waveform with a batch dimension, a channel dimension, and a time dimension.

        The network MUST output a dictionary with the following keys:
            - 'pred_pe': The predicted number of photoelectrons.
            - 'pred_t': The predicted time of the flash.
            - 'pred_c': The predicted confidence of the flash.
            - 'pred_pe_weighted': The predicted number of photoelectrons weighted by the confidence.

        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        self.simulator = simulator

        self.regularizer = regularizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of waveforms. (batch_size, num_channels, num_ticks)
        :return: A tuple of (model_output, sim_output, target)
        """
        target = x[..., :16_000]
        model_output = self.net.forward_with_aggregation(target)
        sim_output = self.simulator(model_output["aggregated_pe"], relax_cut=True)[..., :16_000]
        return model_output, sim_output, target

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()
        pass

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        pe_dist = batch
        raw_wvfm = self.simulator(pe_dist, relax_cut=False)
        model_output, pred, target = self.forward(raw_wvfm)
        loss = self.criterion(pred, target)
        reg_loss = self.regularizer(model_output) if self.regularizer else 0
        return loss, reg_loss, model_output, pred, target

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, reg_loss, model_output, sim_output, target = self.model_step(batch)

        # update and log metrics
        self.log(
            "train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            f"train/{self.regularizer.__class__.__name__.lower()}_loss", reg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            'train/pred_sum', model_output['pred_pe_weighted'].sum(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            'train/target_sum', target.sum(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log_model_output(model_output, "train")

        return loss #+ reg_loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: Loss between raw and reconstructed waveforms.
        """

            
        loss, reg_loss, model_output, pred, target = self.model_step(batch)

        # update and log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            f"val/{self.regularizer.__class__.__name__.lower()}_loss", reg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log_model_output(model_output, "val")
        self.log(
            'val/pred_sum', model_output['pred_pe_weighted'].sum(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            'val/target_sum', target.sum(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        if batch_idx == 0:
            import matplotlib.pyplot as plt

            self.logger.experiment.log({
                f"val/waveform_predictions_{i}": plot_waveform_predictions(model_output, target, batch_idx=i) for i in range(len(batch))
            })
            self.logger.experiment.log(
                {
                    f"val/waveform_predictions_log_{i}": plot_waveform_predictions(
                        model_output, target, batch_idx=i, log=True
                    )
                    for i in range(len(batch))
                }
            )
            plt.close('all')
        return loss #+ reg_loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log(
        #     "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        # )
        pass

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, reg_loss, model_output, pred, target = self.model_step(batch)

        # update and log metrics
        # self.test_loss(loss)
        # self.test_acc(preds, targets)
        self.log(
            "test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            f"test/{self.regularizer.__class__.__name__.lower()}_loss", reg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log_model_output(model_output, "test")
        return loss #+ reg_loss


    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
        self.logger.experiment.watch(self.net, self.criterion, log_freq=1)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def log_model_output(self, model_output: Dict[str, torch.Tensor], prefix: str) -> None:
        """Log the model output to wandb.

        :param model_output: The model output to log.
        :param prefix: The prefix to use for the log.
        """
        keys = ['pred_pe', 'pred_t']

        pred_c = model_output['pred_c'].float().detach().cpu().numpy()
        threshold_mask = pred_c > 0.5

        for key in keys:
            vals = model_output[key].float().detach().cpu().numpy()
            self.logger.experiment.log(
                {
                    f"{prefix}/{key}_c_thresh": wandb.Histogram(vals[threshold_mask.squeeze()]),
                    f"{prefix}/{key}": vals,

                }
            )
        self.logger.experiment.log(
            {
                f"{prefix}/pred_c": wandb.Histogram(pred_c),
            }
        )




if __name__ == "__main__":
    _ = FlashRecoLitModule(None, None, None, None, None, None, None)
