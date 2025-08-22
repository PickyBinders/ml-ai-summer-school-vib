from lightning.pytorch import LightningModule
import torch
from torch.nn import functional as F
import torchmetrics
from lightning.pytorch.cli import LightningCLI


class MyModel(LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        n_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            *[torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)],
            torch.nn.Linear(hidden_dim, output_dim),
        )
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.classification.Accuracy(task="binary"),
                torchmetrics.classification.Precision(task="binary"),
                torchmetrics.classification.Recall(task="binary"),
                torchmetrics.classification.AveragePrecision(task="binary"),
                torchmetrics.classification.AUROC(task="binary"),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx, split):
        output = self(batch)
        loss = F.binary_cross_entropy_with_logits(output, batch["label"])
        self.log(f"{split}/loss", loss, batch_size=batch["batch_size"], sync_dist=True)
        if split == "train":
            log_output = self.train_metrics(output, batch["label"])
            self.log_dict(log_output, batch_size=batch["batch_size"])
        else:
            log_output = self.val_metrics.update(output, batch["label"])
        return loss

    def on_validation_epoch_end(self):
        output = self.val_metrics.compute()
        self.log_dict(output)
        self.val_metrics.reset()

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")


def main():
    """
    Run with python main.py fit -c config.yaml
    Or in an sbatch script with srun python main.py fit -c config.yaml
    """
    torch.set_float32_matmul_precision("medium")
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
