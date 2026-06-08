import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class SincConvEEG(nn.Module):
    """
    Sinc-convolution per segnali EEG con input:
        x.shape = (batch, 1, channels, time_points)

    Output:
        y.shape = (batch, out_channels, channels, time_out)

    Ogni filtro è un passa-banda parametrizzato da:
        - fl_: frequenza inferiore
        - fh_: frequenza superiore

    NOTE:
    - inizializzazione casuale uniforme in [4, 38] Hz
    - nessun vincolo 4-38 durante il training
    - i filtri vengono applicati solo lungo il tempo
    - gli stessi filtri sono condivisi su tutti i canali EEG
    """

    def __init__(
        self,
        out_channels: int = 9,
        kernel_size: int = 31,
        sample_rate: float = 250,
        freq_init_min: float = 4.0,
        freq_init_max: float = 38.0,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            kernel_size += 1  # kernel dispari per simmetria

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.dilation = dilation

        # -----------------------------
        # Inizializzazione SOLO iniziale in 4-38 Hz
        # -----------------------------
        fl_init = torch.empty(out_channels, 1).uniform_(freq_init_min, freq_init_max)
        fh_init = torch.empty(out_channels, 1).uniform_(freq_init_min, freq_init_max)

        self.fl_ = nn.Parameter(fl_init) # li parametrizzo per renderli addestrabili
        self.fh_ = nn.Parameter(fh_init)

        # -----------------------------
        # Finestra di Hamming
        # -----------------------------
        n_lin = torch.linspace(
            0,
            (self.kernel_size // 2) - 1,
            steps=self.kernel_size // 2
        )
        self.register_buffer(
            "window_",
            0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
        )

        # -----------------------------
        # Asse temporale per metà filtro
        # -----------------------------
        n = (self.kernel_size - 1) // 2
        self.register_buffer(
            "n_",
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, C, T)
        returns: (B, F, C, T_out)
        """
        if x.dim() != 4:
            raise ValueError("Input atteso con shape (batch, 1, channels, time_points).")

        if x.size(1) != 1:
            raise ValueError("La seconda dimensione deve essere 1. Shape attesa: (B, 1, C, T).")

        B, _, C, T = x.shape
        device = x.device
        dtype = x.dtype
        
        # ---------------------------------------------------------
        # NESSUN vincolo 4-38 nel training
        # Uso solo ordinamento per garantire low <= high
        # ---------------------------------------------------------
        low = torch.minimum(self.fl_, self.fh_).to(device=device, dtype=dtype)
        high = torch.maximum(self.fl_, self.fh_).to(device=device, dtype=dtype)

        # Evita banda nulla per stabilità numerica
        eps = 1e-6
        high = torch.where((high - low) < eps, low + eps, high)
        band = (high - low)[:, 0]

        # Buffer sul device/dtype corretti
        n = self.n_.to(device=device, dtype=dtype)          # (1, K//2)
        window = self.window_.to(device=device, dtype=dtype)  # (K//2,)

        # ---------------------------------------------------------
        # Costruzione dei filtri sinc band-pass
        # ---------------------------------------------------------
        f_times_t_low = low * n
        f_times_t_high = high * n

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (n / 2.0)
        ) * window

        band_pass_center = 2 * band.view(-1, 1).to(device=device, dtype=dtype)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right],
            dim=1
        )

        # Normalizzazione per banda
        band_pass = band_pass / (2 * band[:, None].to(device=device, dtype=dtype))

        # filters shape: (F, 1, K)
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        # ---------------------------------------------------------
        # Applico i filtri lungo il tempo, indipendentemente per ogni canale EEG
        # x: (B, 1, C, T) -> (B*C, 1, T)
        # ---------------------------------------------------------
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * C, 1, T)

        y = F.conv1d(
            x_reshaped,
            filters,
            stride=self.stride,
            padding=self.kernel_size//2,
            dilation=self.dilation,
            bias=None,
        )  # (B*C, F, T_out)

        T_out = y.size(-1)

        # (B*C, F, T_out) -> (B, F, C, T_out)
        y = y.view(B, C, self.out_channels, T_out).permute(0, 2, 1, 3).contiguous()

        return y
    
###############################################################
##################### MSVTNet #####################
###############################################################


class TSConv(nn.Sequential):
    def __init__(self, inCh, nCh, F, C1, C2, D, P1, P2, Pc) -> None:
        super().__init__(
            nn.Conv2d(inCh, F, (1, C1), padding='same', bias=False),
            nn.BatchNorm2d(F),
            nn.Conv2d(F, F * D, (nCh, 1), groups=F, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(Pc),
            nn.Conv2d(F * D, F * D, (1, C2), padding='same', groups=F * D, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(Pc)
        )


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x):
        x += self.pe
        return x
        

class Transformer(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model, 
        nhead, 
        ff_ratio, 
        Pt = 0.5, 
        num_layers = 4, 
    ) -> None:
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = PositionalEncoding(seq_len + 1, d_model)

        dim_ff =  d_model * ff_ratio
        self.dropout = nn.Dropout(Pt)
        self.trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, Pt, batch_first=True, norm_first=False #era True ho modificato in False per un warining
        ), num_layers, norm=nn.LayerNorm(d_model))

    def forward(self, x):
        b = x.shape[0]
        x = torch.cat((self.cls_embedding.expand(b, -1, -1), x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        return self.trans(x)[:, 0]


class ClsHead(nn.Sequential):
    def __init__(self, linear_in, cls):
        super().__init__(
            nn.Flatten(),
            nn.Linear(linear_in, cls),
            nn.LogSoftmax(dim=1)
        )


class MSVTNet(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVTNet',
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
        in_ch=9
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        self.in_ch = in_ch
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
            nn.Sequential(
                TSConv(self.in_ch, self.nCh, F[b], C1[b], C2, D, P1, P2, Pc),
                Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17)
            )
            for b in range(len(F))
        ])
        branch_linear_in = self._forward_flatten(cat=False)
        self.branch_head = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], num_classes)
            for b in range(len(F))
        ])

        seq_len, d_model = self._forward_mstsconv().shape[1:3] # type: ignore
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1] # type: ignore
        self.last_head = ClsHead(linear_in, num_classes)

    def _forward_mstsconv(self, cat = True):
        x = torch.randn(1, self.in_ch, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        if cat:
            x = torch.cat(x, dim=2)
        return x

    def _forward_flatten(self, cat = True):
        x = self._forward_mstsconv(cat)
        if cat:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = [_.flatten(start_dim=1, end_dim=-1) for _ in x]
        return x

    def forward(self, x):
        x = [tsconv(x) for tsconv in self.mstsconv]
        bx = [branch(x[idx]) for idx, branch in enumerate(self.branch_head)]
        x = torch.cat(x, dim=2)
        x = self.transformer(x)
        x = self.last_head(x)
        if self.b_preds:
            return x, bx
        else:
            return x

###############################################################
##################### SMANet ##################################
###############################################################
class SafeLog(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))

class STCNN(nn.Sequential):
    def __init__(self, inCh, nCh, F, C1, P1) -> None:
        super().__init__(
            nn.Conv2d(inCh, F, (1, C1), bias=False),
            nn.Conv2d(F, F, (nCh, 1), bias=False),
            nn.BatchNorm2d(F),
            nn.AvgPool2d((1, P1), stride=15),
            SafeLog(1e-6),
        )

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class Classification(nn.Sequential):
    def __init__(self, linear_in, cls):
        super().__init__(
            nn.Flatten(),
            nn.Linear(linear_in, cls),
            nn.Softmax(dim=1)
        )


class SMANet(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVTNet',
        F = [5, 5, 5], C1 = [7, 9, 11], P1 = 85, in_ch=36
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.in_ch = in_ch
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.sincnet = SincConvEEG(18, 31)
        self.pointwise = nn.Conv2d(18, in_ch, (1,1))
        self.mbstcnn = nn.ModuleList([
            STCNN(self.in_ch, self.nCh, F[b], C1[b], P1) for b in range(len(F))
        ])
        
        self.eca_module = eca_layer(sum(F))

        linear_in = self._forward_flatten().shape[0]
        self.last_head = ClsHead(linear_in, num_classes)

    def _forward_flatten(self, cat = True):
        x = torch.randn(1, self.in_ch, self.nCh, self.nTime)
        x = [stcnn(x) for stcnn in self.mbstcnn]
        x = torch.cat(x, dim=1)
        x = self.eca_module(x)
        return x.flatten()

    def forward(self, x):
        x = self.sincnet(x)
        x = self.pointwise(x)
        x = [stcnn(x) for stcnn in self.mbstcnn]
        x = torch.cat(x, dim=1)
        x = self.eca_module(x)
        x = self.last_head(x)
        return x


###############################################################
##################### SMANet ##################################
###############################################################

class SincMSVTNet(nn.Sequential):
    def __init__(self, model_name_prefix='SincMSVTNet', num_classes=2, samples=1000, channels=3, b_preds=True) -> None:
        self.model_name_prefix = model_name_prefix
        super().__init__(
            SincConvEEG(),
            MSVTNet(in_ch=9, model_name_prefix=model_name_prefix, num_classes=num_classes, 
                    samples=samples, channels=channels, b_preds=b_preds)
        )