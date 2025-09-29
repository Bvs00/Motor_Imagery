import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
import math


################################################### LMDA #######################################################
class EEGDepthAttention(nn.Module):
    """
    Build EEG Depth Attention module.
    :arg
    C: num of channels
    W: num of time samples
    k: learnable kernel size
    """
    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)  # original kernel k
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        """
        :arg
        """
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)

        return y * self.C * x


class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    """
    def __init__(self, channels=22, samples=1125, num_classes=4, depth=9, kernel=75, channel_depth1=24, channel_depth2=9,
                ave_depth=1, avepool=5, model_name_prefix='LMDANet'):
        super(LMDA, self).__init__()
        self.model_name_prefix = model_name_prefix
        self.ave_depth = ave_depth
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, channels), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)
        # nn.init.kaiming_normal_(self.channel_weight.data, nonlinearity='relu')
        # nn.init.normal_(self.channel_weight.data)
        # nn.init.constant_(self.channel_weight.data, val=1/channels)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )
        # self.avgPool1 = nn.AvgPool2d((1, 24))
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(channels, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )

        out = torch.ones((1, 1, channels, samples))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        N, C, H, W = out.size()

        self.depthAttention = EEGDepthAttention(W, C, k=7)

        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape
        print('In ShallowNet, n_out_time shape: ', n_out_time)
        self.classifier = nn.Linear(n_out_time[-1]*n_out_time[-2]*n_out_time[-3], num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)
        x_time = self.time_conv(x)  # batch, depth1, channel, samples_
        x_time = self.depthAttention(x_time)  # DA1

        x = self.chanel_conv(x_time)  # batch, depth2, 1, samples_
        x = self.norm(x)

        features = torch.flatten(x, 1)
        cls = self.classifier(features)
        return cls


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    """
    Args:
        chunk_size (int): Number of data points included in each EEG chunk, i.e., :math:`T` in the paper. (default: :obj:`151`)
        num_electrodes (int): The number of electrodes, i.e., :math:`C` in the paper. (default: :obj:`60`)
        F1 (int): The filter number of block 1, i.e., :math:`F_1` in the paper. (default: :obj:`8`)
        F2 (int): The filter number of block 2, i.e., :math:`F_2` in the paper. (default: :obj:`16`)
        D (int): The depth multiplier (number of spatial filters), i.e., :math:`D` in the paper. (default: :obj:`2`)
        num_classes (int): The number of classes to predict, i.e., :math:`N` in the paper. (default: :obj:`2`)
        kernel_1 (int): The filter size of block 1. (default: :obj:`64`)
        kernel_2 (int): The filter size of block 2. (default: :obj:`64`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.25`)
    """
    def __init__(self,
                 samples: int = 151,
                 channels: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64, #64
                 kernel_2: int = 16, #16
                 dropout: float = 0.25,
                 model_name_prefix='EEGNet'):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = samples
        self.num_classes = num_classes
        self.num_electrodes = channels
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.channel_weight = nn.Parameter(torch.randn(9, 1, self.num_electrodes), requires_grad=True)
        self.model_name_prefix = model_name_prefix

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.lin = nn.Linear(self.feature_dim(), num_classes, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return self.F2 * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 60, 151]`. Here, :obj:`n` corresponds to the batch size, :obj:`60` corresponds to :obj:`num_electrodes`, and :obj:`151` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        """
        # x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)

        return x

#################################### AUTOENCODER ##########################################

class EncoderEEGNet(nn.Module):
    """
    Args:
        chunk_size (int): Number of data points included in each EEG chunk, i.e., :math:`T` in the paper. (default: :obj:`151`)
        num_electrodes (int): The number of electrodes, i.e., :math:`C` in the paper. (default: :obj:`60`)
        F1 (int): The filter number of block 1, i.e., :math:`F_1` in the paper. (default: :obj:`8`)
        F2 (int): The filter number of block 2, i.e., :math:`F_2` in the paper. (default: :obj:`16`)
        D (int): The depth multiplier (number of spatial filters), i.e., :math:`D` in the paper. (default: :obj:`2`)
        num_classes (int): The number of classes to predict, i.e., :math:`N` in the paper. (default: :obj:`2`)
        kernel_1 (int): The filter size of block 1. (default: :obj:`64`)
        kernel_2 (int): The filter size of block 2. (default: :obj:`64`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.25`)
    """
    def __init__(self,
                 num_electrodes: int = 3,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 kernel_1: int = 64, #64
                 kernel_2: int = 16, #16
                 dropout: float = 0.25):
        super(EncoderEEGNet, self).__init__()
        self.num_electrodes = num_electrodes
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.channel_weight = nn.Parameter(torch.randn(9, 1, self.num_electrodes), requires_grad=True)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = torch.permute(x, (0,2,1))
        return x
    
class EEGNetConformer(nn.Sequential):
    def __init__(self, emb_size=16, num_heads=4, depth=2, num_classes=2, model_name_prefix='EEGConformer', samples=1000, channels=3):
        super().__init__(

            EncoderEEGNet(),
            PositionalEncoding(d_model=emb_size),
            TransformerEncoder(depth, emb_size, num_heads=num_heads),
            Classifier_EEGNetConformer(num_classes)
        )
        self.model_name_prefix = model_name_prefix

class Classifier_EEGNetConformer(nn.Sequential):
    def __init__(self, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(496, 32),
            nn.ELU(),
            nn.Dropout(0.6),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class DecoderEEGNet(nn.Module):
    def __init__(self, num_electrodes, F1, F2, D, kernel_1, kernel_2, dropout):
        super(DecoderEEGNet, self).__init__()
        self.num_electrodes = num_electrodes
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        
        self.deblock1 = nn.Sequential(
            nn.Upsample(size=(1, 251), mode='nearest'),  # Inverso di AvgPool2d(1,8)
            nn.ELU(),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ConvTranspose2d(self.F2, self.F1 * self.D, 1, stride=1, padding=(0, 0), groups=1, bias=False),
            nn.ConvTranspose2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernel_2),
                                stride=1, padding=(0, self.kernel_2 // 2),
                                groups=self.F1 * self.D, bias=False),
            nn.Dropout(p=dropout)
        )

        self.deblock2 = nn.Sequential(
            nn.Upsample(size=(1, 1001), mode='nearest'),
            nn.ELU(),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ConvTranspose2d(self.F1 * self.D, self.F1, (self.num_electrodes, 1),
                                stride=1, padding=(0, 0), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ConvTranspose2d(self.F1, 1, (1, self.kernel_1), stride=1, 
                                padding=(0, self.kernel_1 // 2), bias=False)
        )

    def forward(self, x):
        x = self.deblock1(x)
        x = self.deblock2(x)
        return x


class AutoEncoderEEGNet(nn.Module):
    def __init__(self, F1=8, F2=16, D=2, kernel_1=64, kernel_2=16, dropout=0.25, model_name_prefix="AutoencoderEEGNet", num_electrodes=3):
        super(AutoEncoderEEGNet, self).__init__()
        self.encoder = EncoderEEGNet(num_electrodes, F1, F2, D, kernel_1, kernel_2, dropout)
        self.decoder = DecoderEEGNet(num_electrodes, F1, F2, D, kernel_1, kernel_2, dropout)
        self.model_name_prefix = model_name_prefix

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class EEGNet_Pretrained(nn.Module):
    def __init__(self, encoder, samples: int = 151, channels: int = 60, num_classes=2, model_name_prefix="AutoencoderEEGNet"):
        super(EEGNet_Pretrained, self).__init__()
        self.num_electrodes = channels
        self.chunk_size = samples
        self.model_name_prefix = model_name_prefix

        self.encoder = encoder
        self.lin = nn.Linear(self.feature_dim(), num_classes, bias=False)
    
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.encoder(mock_eeg)

        return mock_eeg.shape[1] * mock_eeg.shape[3]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 60, 151]`. Here, :obj:`n` corresponds to the batch size, :obj:`60` corresponds to :obj:`num_electrodes`, and :obj:`151` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        """
        # x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)

        return x






##################################################################################################################

class CNN_LSTM(nn.Module):

    def __init__(self,samples: int = 151,
                 channels: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25,
                 model_name_prefix='EEGNet',
                 hidden_size=128,
                 bidirectional=False,
                 num_layers=1):
        super(CNN_LSTM, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = samples
        self.num_classes = num_classes
        self.num_electrodes = channels
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.channel_weight = nn.Parameter(torch.randn(9, 1, self.num_electrodes), requires_grad=True)
        self.model_name_prefix = model_name_prefix
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers


        self.block1 = nn.Sequential(
            nn.Conv2d(9, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.lstm_block = nn.LSTM(input_size=self.F2, hidden_size=self.hidden_size, bidirectional=self.bidirectional, batch_first=True, num_layers=self.num_layers)
        self.lin = nn.Linear(self.hidden_size*2 if self.bidirectional else self.hidden_size, num_classes)
        self.linear = nn.Linear(self.feature_dim(), num_classes)


    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 9, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return self.F2 * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 60, 151]`. Here, :obj:`n` corresponds to the batch size, :obj:`60` corresponds to :obj:`num_electrodes`, and :obj:`151` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        """
        # breakpoint()
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)
        x = self.block1(x)
        x = self.block2(x)

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm_block(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.lin(lstm_out)

        # # only EEGNet
        # x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)
        # x = self.block1(x)
        # x = self.block2(x)
        # x = x.flatten(start_dim=1)
        # out = self.linear(x)

        # # only LSTM
        # x = x.squeeze(1)
        # x = x.permute(0, 2, 1)
        # lstm_out, _ = self.lstm_block(x)
        # lstm_out = lstm_out[:, -1, :]
        # out = self.lin(lstm_out)

        return out


class DNN(nn.Module):
    def __init__(self, model_name_prefix="DNN", num_classes: int = 2, samples=1001, channels=3):
        super(DNN, self).__init__()
        self.num_classes = num_classes
        self.model_name_prefix = model_name_prefix
        self.feed_forward = nn.Sequential(
            nn.Linear(3003, 64, bias=True),
            nn.ReLU(inplace=True),
            # nn.Linear(2048, 2048, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Linear(2048, 2048, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Linear(2048, 1024, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, 512, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Linear(512, 64, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Linear(256, 128, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 64, bias=True),
            # nn.ReLU(inplace=True),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.num_classes, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.feed_forward(x.squeeze(1).view(x.shape[0], -1))

        return out


################################################## EEGConformer ########################################################

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (3, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # 5 is better than 1
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class EEGConformer_Wout_Attention(nn.Module):
    def __init__(self, emb_size=40, depth=10, num_classes=2, model_name_prefix='EEGConformer', samples=1001, channels=3):
        # self.patch_size = patch_size
        super().__init__()
        self.model_name_prefix=model_name_prefix
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (3, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2440, 2)
        )
        self.flatten=nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=40, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Creiamo una matrice di zeri [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # Creiamo un vettore di posizioni [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calcoliamo i divisori della formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Applichiamo seno e coseno
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Aggiungiamo una dimensione batch (necessario per operare con batch di input)
        pe = pe.unsqueeze(0)

        # Registriamo il tensore come buffer (non verrà aggiornato durante il training)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, num_heads=5):
        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.cov = nn.Sequential(
            nn.Conv1d(190, 1, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.clshead_fc = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)

        return out


# ! Rethink the use of Transformer for EEG signal
class EEGConformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=10, num_classes=2, model_name_prefix='EEGConformer', samples=1001, channels=3):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, num_classes)
        )
        self.model_name_prefix = model_name_prefix


class EEGConformerPositional(nn.Sequential):
    def __init__(self, emb_size=40, depth=10, num_classes=2, model_name_prefix='EEGConformer_Positional', samples=1001, channels=3):
        super().__init__(

            PatchEmbedding(emb_size),
            PositionalEncoding(emb_size, 61),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, num_classes)
        )
        self.model_name_prefix = model_name_prefix


#########################################################
################ SMALL ENCODER ##########################
#########################################################

# SENet (Squeeze-and-Excitation Network) per l'attenzione sui canali
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


# **Modulo di Filtro Temporale**
class TemporalFiltering(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, kernel_size=(1, 128)):
        super(TemporalFiltering, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.senet = SENet(out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)  # Convoluzione temporale
        x = self.senet(x)  # SENet per attenzione sui canali
        x = self.batch_norm(x)  # Batch Norm
        return x  # Output: (8, C, T)


# **Modulo di Filtro Spaziale**
class SpatialFiltering(nn.Module):
    def __init__(self, in_channels=8, out_channels=16):
        super(SpatialFiltering, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), groups=in_channels)
        self.senet = SENet(out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.senet(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        return x  # Output: (16, 1, T/8)


# **Modulo di Compressione delle Feature**
class FeatureCompression(nn.Module):
    def __init__(self, in_channels=16, out_channels=16):
        super(FeatureCompression, self).__init__()
        self.separable_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 32), groups=in_channels)
        self.senet = SENet(out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 16))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.separable_conv(x)
        x = self.senet(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        return x  # Output: (16, 1, T/128)


# **Classificatore Finale**
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 32)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = self.softmax(x)
        return out  # Output: (batch, c)


class EEGEncoder(nn.Module):
    def __init__(self, num_classes=2, model_name_prefix='EEGConformer', samples=1001, channels=3):
        super(EEGEncoder, self).__init__()
        self.model_name_prefix = model_name_prefix
        self.temporal_filtering = TemporalFiltering()
        self.spatial_filtering = SpatialFiltering()
        self.feature_compression = FeatureCompression()
        self.classifier = Classifier(64, num_classes)  # 16 feature finali

    def forward(self, x):
        x = self.temporal_filtering(x)
        x = self.spatial_filtering(x)
        x = self.feature_compression(x)
        x = self.classifier(x)
        return x


######################################################
############ DILATED Convolution #####################

class EEGNetDilated(nn.Module):
    def __init__(self,
                 model_name_prefix,
                 samples: int = 151,
                 channels: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64, #64
                 kernel_2: int = 16, #16
                 dropout: float = 0.25):
        super(EEGNetDilated, self).__init__()
        self.model_name_prefix = model_name_prefix
        self.chunk_size = samples
        self.num_electrodes = channels
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        
        self.channel_weight = nn.Parameter(torch.randn(9, 1, self.num_electrodes), requires_grad=True)

        # self.block1 = nn.Sequential(
        #     nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
        #     nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
        #     Conv2dWithConstraint(self.F1,
        #                          self.F1 * self.D, (self.num_electrodes, 1),
        #                          max_norm=1,
        #                          stride=1,
        #                          padding=(0, 0),
        #                          groups=self.F1,
        #                          bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
        #     nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        # self.block2 = nn.Sequential(
        #     nn.Conv2d(self.F1 * self.D,
        #               self.F1 * self.D, (1, self.kernel_2),
        #               stride=1,
        #               padding=(0, self.kernel_2 // 2),
        #               bias=False,
        #               groups=self.F1 * self.D),
        #     nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
        #     nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
        #     nn.Dropout(p=dropout))
        
        self.block1_dilated = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), dilation=(1,2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))
        
        self.block2_dilated = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      dilation=(1,2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))
        
        # self.lin = nn.Sequential(
        #     nn.Linear(self.feature_dim(), 512),
        #     nn.ReLU(),
        #     # nn.Linear(512, 32),
        #     # nn.ReLU(),
        #     nn.Linear(512, num_classes, bias=False),
        # )
        self.lin = nn.Linear(self.feature_dim(), num_classes, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            # shallow = self.block1(x)
            # shallow = self.block2(shallow)
            
            deep = self.block1_dilated(x)
            deep = self.block2_dilated(deep)
            
            # shallow = shallow.reshape(shallow.shape[0], -1)
            deep = deep.reshape(deep.shape[0], -1)

        # return torch.cat([shallow, deep], dim=1).shape[1]
        return deep.shape[1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shallow = self.block1(x)
        # shallow = self.block2(shallow)
        
        deep = self.block1_dilated(x)
        deep = self.block2_dilated(deep)
        
        # shallow = shallow.reshape(shallow.shape[0], -1)
        deep = deep.reshape(deep.shape[0], -1)
        
        # out = torch.cat([shallow, deep], dim=1)
        
        return self.lin(deep)
    
    
####### RETE ARTICOLO CKRLNet

class TemporalSpatialConvCKRLNet(nn.Module):
    def __init__(self, in_channels=3, num_filters=24):
        super(TemporalSpatialConvCKRLNet, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, (1, 64), padding='same')  # Conv sul tempo
        self.conv2 = nn.Conv2d(num_filters, num_filters, (in_channels, 1), padding=0, groups=num_filters)  # Depthwise conv spaziale
        self.avgpool1 = nn.AvgPool2d((1, 8))  # Pooling temporale
        self.conv3 = nn.Conv2d(num_filters, num_filters, (1, 16), padding='same')
        self.conv4 = nn.Conv2d(num_filters, num_filters, (1, 1), padding=0)
        self.avgpool2 = nn.AvgPool2d((1, 8))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avgpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool2(x)
        return x  # Output: (24,1,T//64)

class CircularLoopBlockCKRLNet(nn.Module):
    def __init__(self, num_filters=24, dilation_levels=4):
        super(CircularLoopBlockCKRLNet, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(num_filters, num_filters, 3, padding=2**i, dilation=2**i, padding_mode='circular') 
            for i in range(dilation_levels)
        ])
        self.residual = nn.Conv1d(num_filters, num_filters, 1)  # Residual connection

    def forward(self, x):
        res = x
        for conv in self.convs:
            x = conv(x)
        x += self.residual(res)  # Add residual connection
        return x  # Output: (24, T//64)

class EncoderCKRLNet(nn.Module):
    def __init__(self, num_filters=24):
        super(EncoderCKRLNet, self).__init__()
        num_filters_out = 2*num_filters
        self.conv = nn.Conv2d(num_filters, num_filters_out, (2, 1))
        self.bn = nn.BatchNorm2d(num_filters_out)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x  # Output: (48, T//64)
    
class CKRLNet(nn.Module):
    def __init__(self, model_name_prefix, num_classes=2, samples=1000, channels=3):
        super(CKRLNet, self).__init__()
        self.model_name_prefix = model_name_prefix
        self.temporal_spatial = TemporalSpatialConvCKRLNet(channels)
        self.circular_block = CircularLoopBlockCKRLNet()
        self.encoder = EncoderCKRLNet()
        self.lin = nn.Sequential(
            nn.Linear(720, num_classes)
        )
        
    def forward(self, x):
        shallow_features = self.temporal_spatial(x)
        deep_features = self.circular_block(shallow_features.squeeze(2))  # Rimuove la seconda dimensione
        deep_features = deep_features.unsqueeze(2)  # Riaggiunge la dimensione per la concatenazione
        features = torch.cat([shallow_features, deep_features], dim=2)  # Concatenazione delle feature
        features = self.encoder(features)
        features = features.reshape(features.shape[0], -1)
        return self.lin(features)
        
        
        
################################### EEGATTENTIONET ##################################


#########################################################
################ SSCL_CSD ##########################
#########################################################

# SENet (Squeeze-and-Excitation Network) per l'attenzione sui canali
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

# **Modulo di Filtro Temporale**
class TemporalFiltering(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, kernel_size=(1, 128)):
        super(TemporalFiltering, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.senet = SENet(out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)  # Convoluzione temporale
        x = self.senet(x)  # SENet per attenzione sui canali
        x = self.batch_norm(x)  # Batch Norm
        return x  # Output: (8, C, T)

# **Modulo di Filtro Spaziale**
class SpatialFiltering(nn.Module):
    def __init__(self, in_channels=8, out_channels=16):
        super(SpatialFiltering, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), groups=in_channels)
        self.senet = SENet(out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.senet(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        return x  # Output: (16, 1, T/8)

# **Modulo di Compressione delle Feature**
class FeatureCompression(nn.Module):
    def __init__(self, in_channels=16, out_channels=16):
        super(FeatureCompression, self).__init__()
        self.separable_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 32), groups=in_channels)
        self.senet = SENet(out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 16))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.separable_conv(x)
        x = self.senet(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        return x  # Output: (16, 1, T/128)

# **Classificatore Finale**
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 32)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = self.softmax(x)
        return x, out # Output: (batch, c)

class SSCL_CSD(nn.Module):
    def __init__(self, model_name_prefix="Encoder_Small", num_classes=2, samples=1000, channels=3):
        super(SSCL_CSD, self).__init__()
        self.model_name_prefix = model_name_prefix
        self.temporal_filtering = TemporalFiltering()
        self.spatial_filtering = SpatialFiltering()
        self.feature_compression = FeatureCompression()
        self.classifier = Classifier(64, num_classes)  # 16 feature finali

    def forward(self, x):
        x = self.temporal_filtering(x)
        x = self.spatial_filtering(x)
        x = self.feature_compression(x)
        x, out = self.classifier(x)
        return out



################################ CTNET #################################################
class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, number_channel=22, emb_size=16):
        super().__init__()
        f2 = D*f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
                    
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        return x
    
########################################################################################
# The Transformer code is based on this github project and has been fine-tuned: 
#    https://github.com/eeyhsong/EEG-Conformer
########################################################################################
    
class MultiHeadAttention_CTNet(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    


# PointWise FFN
class FeedForwardBlock_CTNet(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )



class ClassificationHead_CTNet(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_number, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        
        return out


class ResidualAdd_CTNet(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        
        out = self.layernorm(self.drop(res)+x_input)
        return out

class TransformerEncoderBlock_CTNet(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd_CTNet(nn.Sequential(
                MultiHeadAttention_CTNet(emb_size, num_heads, drop_p),
                ), emb_size, drop_p),
            ResidualAdd_CTNet(nn.Sequential(
                FeedForwardBlock_CTNet(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                ), emb_size, drop_p)
            
            )    
        
        
class TransformerEncoder_CTNet(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock_CTNet(emb_size, heads) for _ in range(depth)])




class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=3,
                 f1 = 20,
                 kernel_size = 64,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 **kwargs):
        super().__init__(
            PatchEmbeddingCNN(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2, 
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size),
        )


# learnable positional embedding module        
class PositioinalEncoding_CTNet(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
    def forward(self, x): # x-> [batch, embedding, length]
        x = x + self.encoding[:, :x.shape[1], :].cuda()
        return self.dropout(x)        
        
   
        
# CTNet       
class CTNet(nn.Module):
    def __init__(self, heads=2, 
                 emb_size=16,
                 depth=6, 
                 database_type='A', 
                 eeg1_f1 = 8,
                 eeg1_kernel_size = 64,
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
        return out
    
    
################ PATCHEMBEDDING ######################
class PatchEmbeddingNet(nn.Module):
    def __init__(self, f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, channels=3, num_classes=2, model_name_prefix="PatchEmbeddingNet", samples=1000):
        super().__init__()
        f2 = D*f1
        self.channels = channels
        self.samples = samples
        self.model_name_prefix = model_name_prefix
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (channels, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
                    
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        self.classification = nn.Linear(self.num_features_linear(), num_classes)
    
    def num_features_linear(self):
        x = torch.ones((1, 1, self.channels, self.samples))
        x = self.cnn_module(x)
        x = self.projection(x)
        return x.shape[-1]*x.shape[-2]
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        x = x.flatten(start_dim=1)
        x = self.classification(x)
        return x
    
    

#################################################################################################################################
#################################################            MSVTNet            #################################################
#################################################################################################################################


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


#################### CSETNet ####################
# SENet (Squeeze-and-Excitation Network) per l'attenzione sui canali
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
    
class PatchSEEmbedding(nn.Module):
    def __init__(self, f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, number_channel=22, emb_size=16):
        super().__init__()
        f2 = D*f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
            # squeeze-and-excitation
            SENet(f2),
            # 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
                    
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        return x


class BranchEEGSENetTransformer(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=3,
                 f1 = 20,
                 kernel_size = 64,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 **kwargs):
        super().__init__(
            PatchSEEmbedding(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2, 
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size),
        )
        
# CTNet       
class CSETNet(nn.Module):
    def __init__(self, heads=2, 
                 emb_size=16,
                 depth=6, 
                 database_type='A', 
                 eeg1_f1 = 8,
                 eeg1_kernel_size = 64,
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
        self.cnn = BranchEEGSENetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
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
        return out
    
    
#################### MSVTSENet ####################

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
    
################################# SuperCTNet ######################
class SuperPatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=16, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3, number_channel=22, emb_size=40):
        super().__init__()
        f2 = D*f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
            
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),
            
            #nn.AvgPool2d((1, 4)),
            nn.AvgPool2d((1, 4)), # output fixed length 16
            nn.Dropout(dropout_rate),

        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape  # input shape = [batch size, feature channel 1, electrode channel (22 for 2a, 3 for 2b), sample point 1000]
        x = self.cnn_module(x)
        x = self.projection(x)
        return x

class BranchSuperEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=3,
                 f1 = 20,
                 kernel_size = 64,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 **kwargs):
        super().__init__(
            SuperPatchEmbeddingCNN(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2, 
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size),
        )
        
# CTNet       
class SuperCTNet(nn.Module):
    def __init__(self, heads=2, 
                 emb_size=16,
                 depth=6, 
                 database_type='A', 
                 eeg1_f1 = 8,
                 eeg1_kernel_size = 64,
                 eeg1_D = 2,
                 eeg1_pooling_size1 = 2,
                 eeg1_pooling_size2 = 3,
                 eeg1_dropout_rate = 0.25,
                 flatten_eeg1 = 656,
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
        self.cnn = BranchSuperEEGNetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
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
        return out