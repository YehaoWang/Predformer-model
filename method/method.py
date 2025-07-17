import math
import torch
import torch.nn as nn
import lightning
from typing import List
from .attention import GatedAttentionUnit


class UlvaFormer(lightning.LightningModule):
    def __init__(self,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layer: int = 4,
                 r_forward: int = 4,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 data_back: List[int] = None,
                 data_pred: List[int] = None,
                 lr: float = 1e-3,
                 lr_scheduler='cosine',
                 attn_bias: bool = False,
                 ):
        super(UlvaFormer, self).__init__()
        # Parameters:
        assert lr_scheduler in ['cosine', 'onecycle']
        self.data_back = data_back if data_back is not None else [7, 8, 12]  # [T,N,C]
        self.data_pred = data_pred if data_pred is not None else [7, 8, 1]  # [T,N,target]
        self.lr = lr
        self.lr_scheduler = lr_scheduler

        # Lightning Framework:
        self.example_input_array = torch.rand(1, *self.data_back)
        self.save_hyperparameters()

        # Criterion:
        self.criterion = nn.MSELoss(reduction='mean')

        self.encoder = Encoder(d_model=d_model, data_back=self.data_back)
        self.attn = nn.ModuleList(
            nn.ModuleList([
                GatedAttentionUnit(d_model, n_heads, dropout, drop_path, r_forward, attn_bias),
                GatedAttentionUnit(d_model, n_heads, dropout, drop_path, r_forward, attn_bias)
            ]) for i in range(n_layer)
        )
        self.decoder = Decoder(d_model=d_model, data_pred=self.data_pred)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        B, T, N, C = x.shape
        x = self.encoder(x)  # B,T,N,D
        attn_map_t_list = []
        attn_map_s_list = []

        for (t_attn, s_attn) in self.attn:
            x = x.permute(0, 2, 1, 3).flatten(0, 1)
            x, attn_map_t = t_attn(x, return_attn=True)
            x = x.reshape(B, N, T, -1).permute(0, 2, 1, 3).flatten(0, 1)
            x, attn_map_s = s_attn(x, return_attn=True)
            x = x.reshape(B, T, N, -1)
            attn_map_t_list.append(attn_map_t)
            attn_map_s_list.append(attn_map_s)

        x = self.decoder(x)

        if return_attn:
            return x, attn_map_t_list, attn_map_s_list
        else:
            return x

    def shared_step(self, batch, return_attn: bool = False):
        seq_back, seq_true = batch
        if return_attn:
            seq_pred, attn_map_t, attn_map_s = self(seq_back, return_attn)
        else:
            seq_pred = self(seq_back, return_attn)
        seq_pred=seq_pred.squeeze(-1)

        loss = self.criterion(seq_pred, seq_true)

        if return_attn:
            return loss, seq_pred, seq_true, attn_map_t, attn_map_s
        else:
            return loss, seq_pred, seq_true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, seq_pred, seq_true,  attn_map_t, attn_map_s = self.shared_step(batch, True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return {  # Callback: LogValidationMetric need 'seq_pred' and 'seq_true'
            'loss': loss,
            'seq_pred': seq_pred,
            'seq_true': seq_true,
            'attn_map_t': attn_map_t,
            'attn_map_s': attn_map_s,
        }

    def test_step(self, batch, batch_idx):
        loss, seq_pred, seq_true, attn_map_t, attn_map_s = self.shared_step(batch, True)
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return {  # Callback: LogValidationMetric need 'seq_pred' and 'seq_true'
            'loss': loss,
            'seq_pred': seq_pred,
            'seq_true': seq_true,
            'attn_map_t': attn_map_t,
            'attn_map_s': attn_map_s,
        }


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)  # AdamW Optimizer
        if self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=5e-8
            )
        elif self.lr_scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.lr,
                epochs=self.trainer.max_epochs,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                div_factor=25,
                final_div_factor=10000,
            )
        else:
            scheduler = None
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


class Decoder(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 data_pred: List[int] = None):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(in_features=d_model, out_features=data_pred[-1], bias=True)
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        x = self.linear(x)  # B,T,N,C
        # x = self.sigmoid(x)

        return x  # B,T,N,C


class Encoder(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 data_back: List[int] = None):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(in_features=data_back[-1], out_features=d_model, bias=True)
        self.norm_activation = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.SiLU()
        )
        self.pos_encode = SpatiotemporalPosEncoding2D(d_model=d_model, T=data_back[0], N=data_back[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        x = self.linear(x)  # B,T,N,D
        x = self.norm_activation(x)  # B,T,N,D
        x = self.pos_encode(x)
        return x  # B,T,N,D


class SpatiotemporalPosEncoding2D(nn.Module):
    def __init__(self, d_model: int, T: int, N: int):
        super(SpatiotemporalPosEncoding2D, self).__init__()
        assert d_model % 2 == 0, "d_model must be divisible by 4 for separate T, H, W encoding."

        self.d_n = d_model // 2
        self.d_t = d_model // 2

        self.pe_t = self.generate_sinusoidal_encoding(T, self.d_t)
        self.pe_n = self.generate_sinusoidal_encoding(N, self.d_n)
        self.register_buffer("pe", self.combine_encodings(self.pe_t, self.pe_n))

    def generate_sinusoidal_encoding(self, length, d_model):
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)  # Shape: [length, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # Shape: [d_model/2]
        encoding = torch.zeros((length, d_model))  # Shape: [length, d_model]
        encoding[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        encoding[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        return encoding

    def combine_encodings(self, pe_t, pe_n):
        T, d_t = pe_t.shape
        HW, d_n = pe_n.shape

        pe_t = pe_t.unsqueeze(1).expand(T, HW, d_t)  # [T, HW, d_t]
        pe_n = pe_n.unsqueeze(0).expand(T, HW, d_n)  # [T, HW, d_hw]

        pe = torch.cat([pe_t, pe_n], dim=-1)  # [T, HW, d_t + d_hw]
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # [1, D, T, HW]
        return pe

    def forward(self, x):
        B, T, N, D = x.shape
        return x + self.pe.permute(0, 2, 3, 1)


if __name__ == '__main__':
    # Example usage with dummy data
    batch_size = 2
    time_steps = 7
    num_nodes = 8
    num_features = 12
    d_model = 64  # Reduced d_model for faster testing

    # Create dummy input tensor
    dummy_input = torch.randn(batch_size, time_steps, num_nodes, num_features)

    # 3. Test UlvaFormer (with reduced parameters for quick testing)
    ulva_former = UlvaFormer(d_model=d_model, data_back=[time_steps, num_nodes, num_features],
                             data_pred=[3, num_nodes, 1])
    model_output = ulva_former(dummy_input)
    print("UlvaFormer Output Shape:", model_output.shape)
    assert model_output.shape == (batch_size, time_steps, num_nodes, d_model), "Incorrect shape from UlvaFormer"

    print("All tests passed!")
