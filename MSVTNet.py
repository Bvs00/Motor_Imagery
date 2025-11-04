#################################################################################################################################
#################################################            MSVTNet            #################################################
#################################################################################################################################
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
import math

class TSConv(nn.Sequential):
    def __init__(self, nCh, F, C1, C2, D, P1, P2, Pc) -> None:
        super().__init__(
            nn.Conv2d(1, F, (1, C1), padding='same', bias=False),
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
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
            nn.Sequential(
                TSConv(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc),
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
        x = torch.randn(1, 1, self.nCh, self.nTime)
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

#################### MSVTSENet ####################

class SENet(nn.Module):
    def __init__(self, channels, reduction=2):
        super(SENet, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc1 = nn.Linear(channels, channels // reduction)  # Riduzione dimensionale
        self.fc2 = nn.Linear(channels // reduction, channels)  # Espansione dimensionale
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, feature_maps, _, _ = x.shape
        y = self.global_avg_pool(x).view(batch, feature_maps)  # Global Average Pooling
        y = F.relu(self.fc1(y))  # ReLU dopo la riduzione dimensionale
        y = self.sigmoid(self.fc2(y)).view(batch, feature_maps, 1, 1)  # Sigmoid e reshape
        return x * y  # Applicazione dei pesi ai canali originali


class MSVTSENet(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVTNet',
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
                TSConv(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc)
            for b in range(len(F))
        ])
        self.rearrange = Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17)
        self.se_module = SENet(D*sum(F))
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
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        x = [self.rearrange(x_i) for x_i in x]
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
        x = torch.cat(x, dim=1)
        x = self.se_module(x)
        x = self.rearrange(x)
        x = self.transformer(x)
        x = self.last_head(x)
        if self.b_preds:
            return x, bx
        else:
            return x


################################### MSSEVTNet #######################################
class TSSEConv(nn.Sequential):
    def __init__(self, nCh, F, C1, C2, D, P1, P2, Pc) -> None:
        super().__init__(
            nn.Conv2d(1, F, (1, C1), padding='same', bias=False),
            nn.BatchNorm2d(F),
            nn.Conv2d(F, F * D, (nCh, 1), groups=F, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(Pc),
            nn.Conv2d(F * D, F * D, (1, C2), padding='same', groups=F * D, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            SENet(F * D),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(Pc)
        )

class MSSEVTNet(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSSEVTNet',
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
            nn.Sequential(
                TSSEConv(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc),
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
        x = torch.randn(1, 1, self.nCh, self.nTime)
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

################################### MSSEVTSENet #######################################

class MSSEVTSENet(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVTNet',
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
                TSSEConv(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc)
            for b in range(len(F))
        ])
        self.rearrange = Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17)
        self.se_module = SENet(D*sum(F))
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
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        x = [self.rearrange(x_i) for x_i in x]
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
        x = torch.cat(x, dim=1)
        x = self.se_module(x)
        x = self.rearrange(x)
        x = self.transformer(x)
        x = self.last_head(x)
        if self.b_preds:
            return x, bx
        else:
            return x
        
#################### MSVTSE_ChPos_Net ####################
class TSConv_Ch(nn.Module):
    def __init__(self, nCh, F, C1, C2, D, P1, P2, Pc) -> None:
        super().__init__()
        self.channel_weight = nn.Parameter(torch.randn(F, 1, nCh), requires_grad=True) # matrice di pesi addestrabile per enfatizzare i canali Fx1xCh
        
        self.main_network = nn.Sequential(
            nn.Conv2d(F, F, (1, C1), padding='same', bias=False),
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
    
    def forward(self, x):
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)
        return self.main_network(x)

class MSVTSE_ChEmphasis_Net(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVTSE_ChEmphasis_Net',
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
                TSConv_Ch(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc)
            for b in range(len(F))
        ])
        self.rearrange = Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17)
        self.se_module = SENet(D*sum(F))
        branch_linear_in = self._forward_flatten(cat=False)
        self.branch_head = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], num_classes)
            for b in range(len(F))
        ])

        seq_len, d_model = self._forward_mstsconv().shape[1:3] # type: ignore
        print(seq_len)
        print(d_model)
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1] # type: ignore
        self.last_head = ClsHead(linear_in, num_classes)

    def _forward_mstsconv(self, cat = True):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        x = [self.rearrange(x_i) for x_i in x]
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
        x = torch.cat(x, dim=1)
        x = self.se_module(x)
        x = self.rearrange(x)
        x = self.transformer(x)
        x = self.last_head(x)
        if self.b_preds:
            return x, bx
        else:
            return x


################################### MSVT_SE_Net #######################################
class TSConv_SE(nn.Sequential):
    def __init__(self, nCh, F, C1, C2, D, P1, P2, Pc) -> None:
        super().__init__(
            nn.Conv2d(1, F, (1, C1), padding='same', bias=False),
            SENet(F),
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

class MSVT_SE_Net(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVT_SE_Net',
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
            nn.Sequential(
                TSConv_SE(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc),
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
        x = torch.randn(1, 1, self.nCh, self.nTime)
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
        
        
################################### MSVT_SE_SE_Net #######################################

class MSVT_SE_SE_Net(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVT_SE_SE_Net',
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
            TSConv_SE(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc)
            for b in range(len(F))
        ])
        self.rearrange = Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17
        self.se_module = SENet(D*sum(F))
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
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        x = [self.rearrange(x_i) for x_i in x]
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
        x = torch.cat(x, dim=1)
        x = self.se_module(x)
        x = self.rearrange(x)
        x = self.transformer(x)
        x = self.last_head(x)
        if self.b_preds:
            return x, bx
        else:
            return x
        
################################### MSVT_SE_SE_Net_WoutAux #######################################

class MSVT_SE_SE_Net_WoutAux(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVT_SE_SE_Net_WoutAux',
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
            TSConv_SE(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc)
            for b in range(len(F))
        ])
        self.rearrange = Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17
        self.se_module = SENet(D*sum(F))
        branch_linear_in = self._forward_flatten(cat=False)

        seq_len, d_model = self._forward_mstsconv().shape[1:3] # type: ignore
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1] # type: ignore
        self.last_head = ClsHead(linear_in, num_classes)

    def _forward_mstsconv(self, cat = True):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        x = [self.rearrange(x_i) for x_i in x]
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
        x = torch.cat(x, dim=1)
        x = self.se_module(x)
        x = self.rearrange(x)
        x = self.transformer(x)
        x = self.last_head(x)
        return x