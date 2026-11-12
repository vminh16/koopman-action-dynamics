import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from models.model_builder import build_model
from models.model_builder import NeuralODEModel
from models.model_builder import MLPModel
from training.loss import prediction_loss, encoder_loss_from_pred, encoder_loss_from_true, controllability_loss
from dynamics import SYSTEM_REGISTRY
from my_utils.dataset import load_dataset
from my_utils.dataset import SampledMultiTrajectoryDataset as KoopmanTrajectoryDataset


class NeuralODELitModule(pl.LightningModule):
    def __init__(self, system_config, model_config, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = NeuralODEModel(
            state_dim=system_config["state_dim"],
            control_dim=system_config["umin"].shape[0],
            hidden_dim=model_config.get("hidden_size", 128),
            dt=system_config["dt"],
            ode_solver=model_config.get("ode_solver", "euler")
        )

    def forward(self, x0, u_seq):
        return self.model(x0, u_seq)

    def training_step(self, batch, batch_idx):
        x0, u_seq, x_seq = batch
        x_hat_seq = self(x0, u_seq)
        loss = prediction_loss(x_hat_seq, x_seq)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

class MLPLitModule(pl.LightningModule):
    def __init__(self, system_config, model_config, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLPModel(
            state_dim=system_config["state_dim"],
            control_dim=system_config["umin"].shape[0],
            hidden_dim=model_config.get("hidden_size", 128),
            num_layers=model_config.get("num_layers", 3),
            activation=model_config.get("activation", "tanh"),
        )

    def forward(self, x0, u_seq):
        return self.model(x0, u_seq)

    def training_step(self, batch, batch_idx):
        x0, u_seq, x_seq = batch                # x_seq: [B, T, dx] (真实 1..T)
        x_hat_seq = self(x0, u_seq)             # 预测 [B, T, dx]
        loss = prediction_loss(x_hat_seq, x_seq)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, u_seq, x_seq = batch
        x_hat_seq = self(x0, u_seq)
        loss = prediction_loss(x_hat_seq, x_seq)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x0, u_seq, x_seq = batch
        x_hat_seq = self(x0, u_seq)
        loss = prediction_loss(x_hat_seq, x_seq)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class KoopmanLitModule(pl.LightningModule):
    def __init__(self, system_config, model_config, lr=1e-3):
        super().__init__()
        self.save_hyperparameters('system_config', 'model_config', 'lr')
        self.model = build_model(system_config, model_config)
        self.state_dim = system_config["state_dim"]
        self.koopman_mode = self.hparams.model_config.get("koopman_mode")
        self.lambda_encoder_pred = self.hparams.model_config.get("lambda_encoder_pred", 0.5)
        self.lambda_control = self.hparams.model_config.get("lambda_control", 0.005)
        self.encoder_loss_flag = self.hparams.model_config.get("encoder_loss_flag", True)
        self.control_loss_flag = self.hparams.model_config.get("control_loss_flag", True)
        self.dt = system_config["dt"]


    def forward(self, x0, u_seq):
        return self.model(x0, u_seq)

    def training_step(self, batch, batch_idx):

        # encoder_loss_flag = True  # Renamed from encoder_loss to avoid conflict with function
        # control_loss_flag = False  # Renamed from control_loss

        # lambda_encoder_pred = 0.5
        # lambda_control = 0.01

        x0, u_seq, x_seq = batch
        x0 = x0.squeeze(1)  # squeeze
        x_seq = x_seq.squeeze(2)  # squeeze
        z_seq_pred = self(x0, u_seq)
        x_pred = z_seq_pred[:, :, :self.state_dim]  # [B, T, dx]
        # Initialize losses that might be conditional to avoid UnboundLocalError if flags are False
        ctrl_loss = torch.tensor(0.0, device=self.device)
        encoder_loss_from_xtrue_val = torch.tensor(0.0, device=self.device)  # Renamed for clarity
        encoder_loss_from_xpred_val = torch.tensor(0.0, device=self.device)  # Renamed for clarity

        # Main prediction loss
        pred_loss = prediction_loss(x_pred, x_seq)
        loss = pred_loss.clone()

        if self.encoder_loss_flag:
            encoder_loss_from_xtrue_val = encoder_loss_from_true(
                x_seq, z_seq_pred[:, :, self.state_dim:], self.model.encoder
            )
            loss += encoder_loss_from_xtrue_val

        encoder_loss_from_xpred_val = encoder_loss_from_pred(
            x_pred, z_seq_pred[:, :, self.state_dim:], self.model.encoder
        )
        loss += self.lambda_encoder_pred * encoder_loss_from_xpred_val

        # Controllability loss from Koopman A, B
        # Call build_A() and build_B() methods to get the matrices

        if self.koopman_mode == 'free':
            A = self.model.operator.core.A
            B = self.model.operator.core.B
            ctrl_loss = controllability_loss(A, B, state_dim=self.state_dim, T=0.3, dt=self.dt, method="coupled")
        if self.koopman_mode == 'controllable':
            # A = self.model.operator.core.build_A() # # Call build_A() and build_B() methods to get the matrices
            # B = self.model.operator.core.build_B()
            A, B = self.model.operator.core.get_transformed_AB()
            ctrl_loss = controllability_loss(A, B, state_dim=self.state_dim, T=0.3, dt=self.dt, method="coupled")

        # import pdb; pdb.set_trace()
        if self.control_loss_flag:
            loss += self.lambda_control * ctrl_loss

        self.log("train_total_loss", loss, prog_bar=True)  # Changed name for clarity
        self.log("loss/prediction", pred_loss, prog_bar=True)
        if self.encoder_loss_flag:
            self.log("loss/encoder_true", encoder_loss_from_xtrue_val, prog_bar=True)
        # self.log("loss/encoder_pred", encoder_loss_from_xpred_val, prog_bar=True)
        if self.control_loss_flag:
            self.log("loss/controllability", ctrl_loss, prog_bar=True)
        # from torch.autograd import grad
        # grads = grad(ctrl_loss, [A, B], retain_graph=True, allow_unused=True)
        # print("ctrl_loss grad wrt A:", grads[0])
        # print("ctrl_loss grad wrt B:", grads[1])

        # A_grad = self.model.operator.core.A.grad
        # B_grad = self.model.operator.core.B.grad
        # print("A_free grad:", A_grad)
        # print("B_free grad:", B_grad)
        # for name, param in self.model.encoder.named_parameters():
        #     if param.grad is not None:
        #         print(f"Encoder param: {name}, grad norm: {param.grad.norm().item():.4e}")
        #     else:
        #         print(f"Encoder param: {name}, grad is None")
        return loss

    def configure_optimizers(self):
        # AdamW
        encoder_params = list(self.model.encoder.parameters())
        operator_params = list(self.model.operator.parameters())

        optimizer = torch.optim.AdamW(
            # [
            # {'params': encoder_params, 'lr': self.hparams.lr},  # baseline lr
            # {'params': operator_params, 'lr': self.hparams.lr * 10.0}],  # higher lr for A/B
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-5  #
        )
        # ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_total_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
