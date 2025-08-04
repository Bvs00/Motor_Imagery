import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from utils import available_network, network_factory_methods, available_paradigm, \
    create_tensors, load_normalizations, find_minum_loss
from torch.utils.data import TensorDataset, DataLoader


class GradCAMExtractor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output).to(args.device)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])  # global avg pool
        activations = self.activations[0]  # shape: [C, H, W]

        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]

        cam = torch.sum(activations, dim=0)
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        return cam.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

# ========== VISUALIZZAZIONE CAM + SEGNALE ==========
# def plot_cam_on_signal(cam, signal, title="GradCAM overlay"):
#     """
#     cam: torch.Tensor [1, 1, H, W]
#     signal: torch.Tensor [C, T]
#     """
#     # breakpoint()
#     cam = F.interpolate(cam, size=(1, signal.shape[1]), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
#     time = np.arange(signal.shape[1])

#     plt.figure(figsize=(12, 5))
#     for i in range(signal.shape[0]):
#         plt.plot(time, signal[i] + i*5, label=f"Ch {i+1}")  # offset ogni canale
#     plt.imshow(cam[np.newaxis, :], aspect='auto', extent=[0, signal.shape[1], -5, signal.shape[0]*5],
#                cmap='jet', alpha=0.5, origin='lower')
    
#     plt.title(title)
#     plt.xlabel("Tempo")
#     plt.ylabel("Segnale")
#     plt.legend()
#     plt.colorbar(label="CAM Intensity", orientation='vertical')
#     plt.tight_layout()
#     plt.savefig('prova')
import matplotlib.colors as mcolors

def plot_cam_on_signal_lines(cam, signal, title="GradCAM on Signal"):
    cam = F.interpolate(cam, size=(1, signal.shape[1]), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    signal = signal.cpu().numpy()

    time = np.arange(signal.shape[1])
    fig, ax = plt.subplots(figsize=(12, 5))

    for i in range(signal.shape[0]):
        x = time
        y = signal[i] + i * 5  # offset verticale
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm_cam = cam / (cam.max() + 1e-6)
        colors = plt.cm.jet(norm_cam)
        lc = LineCollection(segments, colors=colors[:-1], linewidth=2)
        ax.add_collection(lc)
        ax.plot([], [], color='black', label=f"Ch {i+1}")

    ax.set_xlim(0, signal.shape[1])
    ax.set_ylim(signal.min() - 2, signal.max() + 2 + (signal.shape[0] - 1) * 5)
    ax.set_title(title)
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Segnale (offset per canale)")

    # Crea ScalarMappable con la stessa normalizzazione di cam
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])  # serve anche se non è associato a un'immagine

    # Colorbar in un nuovo axes a destra
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Intensità CAM')

    ax.legend()
    plt.tight_layout()
    plt.savefig('prova')

# ========== MAIN ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, default='/mnt/datasets/eeg/Dataset_BCI_2b/Signals_BCI_2classes/test_2b_full.npz')
    parser.add_argument("--name_model", type=str, default='PatchEmbeddingNet', help="Name of model that use", choices=available_network)
    parser.add_argument('--saved_path', type=str, default='Results_Black/Results_Z_Score_unique/Results_SegRec/Results_Single/Results_PatchEmbeddingNet_Full_NoBandpass')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--paradigm', type=str, choices=available_paradigm, default='Single')
    args = parser.parse_args()
    
    data_test_tensors, labels_test_tensors = create_tensors(args.test_set)
    loss_list, f1_list, accuracy_list, balanced_accuracy_list, final_results = [], [], [], [], []
    
    for patient in range(len(data_test_tensors)):
        data, labels = data_test_tensors[patient], labels_test_tensors[patient]
        
        saved_path = args.saved_path if args.paradigm=='Cross' else f'{args.saved_path}/Patient_{patient + 1}'
        
        mean, std, min_, max_ = load_normalizations(f'{saved_path}/{args.name_model}')
        
        if mean != None:
            data = (data - mean)/std
        else:
            data = (data - min_)/(max_ - min_)

        # dataset = TensorDataset(data, labels)
        # test_loader = DataLoader(dataset, batch_size=256, num_workers=5)

        best_fold = find_minum_loss(f'{saved_path}/{args.name_model}_seed{args.seed}_validation_log.txt')

        model = (
            network_factory_methods[args.name_model](model_name_prefix=f'{saved_path}/{args.name_model}_seed{args.seed}',
                num_classes=len(np.unique(labels)),
                samples=data.shape[3], channels=data.shape[2])
        )
        model.to(args.device)
        model.load_state_dict(torch.load(f'{saved_path}/{args.name_model}_seed{args.seed}_best_model_fold{best_fold}.pth'))
        model.eval()
        input_tensor = data[0].unsqueeze(1).to(args.device)
        # GradCAM
        target_layer = model.cnn_module[10]
        cam_extractor = GradCAMExtractor(model, target_layer)
        cam = cam_extractor.generate_cam(input_tensor)

        # Visualizzazione
        signal=data[0][0]
        plot_cam_on_signal_lines(cam, signal)