import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F


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
        latent_representation = self.transformer(x)
        x = self.last_head(latent_representation)
        if self.b_preds:
            return x, bx, latent_representation
        else:
            return x, latent_representation

#################### MSVTSENet ####################

class SENet(nn.Module):
    def __init__(self, channels, reduction=2):
        super(SENet, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc1 = nn.Linear(channels, channels // reduction)  # Riduzione dimensionale
        self.fc2 = nn.Linear(channels // reduction, channels)  # Espansione dimensionale
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_attention=False):
        batch, feature_maps, _, _ = x.shape
        y = self.global_avg_pool(x).view(batch, feature_maps)  # Global Average Pooling
        y = F.relu(self.fc1(y))  # ReLU dopo la riduzione dimensionale
        y = self.sigmoid(self.fc2(y)).view(batch, feature_maps, 1, 1)  # Sigmoid e reshape
        if return_attention:
            return x * y, y
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

    def forward(self, x, return_attention=False):
        x = [tsconv(x) for tsconv in self.mstsconv]
        bx = [branch(x[idx]) for idx, branch in enumerate(self.branch_head)]
        x = torch.cat(x, dim=1)
        if return_attention:
            x, se_weights = self.se_module(x, return_attention)
        else:
            x = self.se_module(x)
        x = self.rearrange(x)
        latent_representation = self.transformer(x)
        x = self.last_head(latent_representation)
        if self.b_preds:
            return [x, bx, None, se_weights] if return_attention else [x, bx, latent_representation]
        else:
            return [x, None, se_weights] if return_attention else x, latent_representation


################################### MSVT_SE_Net #######################################
class TSConv_SE(nn.Module):
    def __init__(self, nCh, F, C1, C2, D, P1, P2, Pc):
        super().__init__()

        # --- First stage ---
        self.conv1 = nn.Conv2d(1, F, (1, C1), padding='same', bias=False)
        self.se1 = SENet(F)
        self.bn1 = nn.BatchNorm2d(F)

        # --- Depthwise conv (spatial) ---
        self.depth_conv1 = nn.Conv2d(F, F * D, (nCh, 1), groups=F, bias=False)
        self.bn2 = nn.BatchNorm2d(F * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, P1))
        self.drop1 = nn.Dropout(Pc)

        # --- Temporal depthwise conv ---
        self.depth_conv2 = nn.Conv2d(F * D, F * D, (1, C2),
                                     padding='same', groups=F * D, bias=False)
        self.bn3 = nn.BatchNorm2d(F * D)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, P2))
        self.drop2 = nn.Dropout(Pc)

    def forward(self, x, return_attention=False):
        # First convolution + SE + BN
        x = self.conv1(x)
        if return_attention:
            x, se_weights = self.se1(x, return_attention)
        else:
            x = self.se1(x)
        x = self.bn1(x)

        # Depthwise spatial conv
        x = self.depth_conv1(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Depthwise temporal conv
        x = self.depth_conv2(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return [x, se_weights] if return_attention else x 

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

    def forward(self, x, return_attention):
        if return_attention:
            tsconv_se_out = [tsconv[0](x, return_attention) for tsconv in self.mstsconv]
            x = [tsconv[1](tsconv_se_out[i][0]) for i,tsconv in enumerate(self.mstsconv)]
            se_weights_branches = [branch[1] for branch in tsconv_se_out]
        else:
            x = [tsconv(x) for tsconv in self.mstsconv]
        bx = [branch(x[idx]) for idx, branch in enumerate(self.branch_head)]
        x = torch.cat(x, dim=2)
        latent_representation = self.transformer(x)
        x = self.last_head(latent_representation)
        if self.b_preds:
            return [x, bx, se_weights_branches, None] if return_attention else [x, bx, latent_representation]
        else:
            return [x, se_weights_branches, None] if return_attention else x, latent_representation
        
        
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

    def forward(self, x, return_attention=False):
        if return_attention:
            x = [tsconv(x, return_attention) for tsconv in self.mstsconv]
            se_weights_branches = [branch[1] for branch in x]
            x = [branch[0] for branch in x]
        else:
            x = [tsconv(x) for tsconv in self.mstsconv]
        bx = [branch(x[idx]) for idx, branch in enumerate(self.branch_head)]
        x = torch.cat(x, dim=1)
        if return_attention:
            x, se_weights = self.se_module(x, return_attention)
        else:
            x = self.se_module(x)
        x = self.rearrange(x)
        latent_representation = self.transformer(x)
        x = self.last_head(latent_representation)
        if self.b_preds:
            return [x, bx, se_weights_branches, se_weights] if return_attention else [x, bx, latent_representation]
        else:
            return [x, se_weights_branches, se_weights] if return_attention else [x, latent_representation]


##################################### CTNET ###########################################
from CTNet import BranchEEGNetTransformer, PositioinalEncoding_CTNet, TransformerEncoder_CTNet, ClassificationHead_CTNet
import math
class CTNet(nn.Module):
    def __init__(self, heads=2, 
                 emb_size=16,
                 depth=6, 
                 database_type='A', 
                 eeg1_f1 = 8,
                 eeg1_kernel_size = 63,
                 eeg1_D = 2,
                 eeg1_pooling_size1 = 8,
                 eeg1_pooling_size2 = 8,
                 eeg1_dropout_rate = 0.25,
                 flatten_eeg1 = 240,
                 num_classes = 2,
                 channels = 3,
                 model_name_prefix="CTNet",
                 **kwargs):
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.number_class, self.number_channel = num_classes, channels
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()
        print('self.number_channel', self.number_channel)
        self.cnn = BranchEEGNetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
                                              f1 = eeg1_f1,
                                              kernel_size = eeg1_kernel_size,
                                              D = eeg1_D,
                                              pooling_size1 = eeg1_pooling_size1,
                                              pooling_size2 = eeg1_pooling_size2,
                                              dropout_rate = eeg1_dropout_rate,
                                              )
        self.position = PositioinalEncoding_CTNet(emb_size, dropout=0.1)
        self.trans = TransformerEncoder_CTNet(heads, depth, emb_size)

        self.flatten = nn.Flatten()
        self.classification = ClassificationHead_CTNet(self.flatten_eeg1 , self.number_class) # FLATTEN_EEGNet + FLATTEN_cnn_module
    
    def forward(self, x):
        cnn = self.cnn(x)

        #  positional embedding
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        
        trans = self.trans(cnn)
        # residual connect
        features = cnn+trans
        
        out = self.classification(self.flatten(features))
        return out, None, self.flatten(features)
        
network_factory_methods = {
    'MSVTNet': MSVTNet,
    'MSVTSENet': MSVTSENet,
    'MSVT_SE_Net': MSVT_SE_Net,
    'MSVT_SE_SE_Net': MSVT_SE_SE_Net,
    'CTNet': CTNet
}