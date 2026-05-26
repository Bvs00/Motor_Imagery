import argparse
import torch
from scipy.signal import freqz
from utils import available_network, find_minum_loss, network_factory_methods
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_model", type=str, default='MSVTNet', help="Name of tensors that use", choices=available_network)
    parser.add_argument('--saved_path', type=str, default='Results_Black/Results_Z_Score_unique/Results_SegRec/Results_Cross/Results_MSVTNet_Full_NoBandpass')
    parser.add_argument('--saved_path_visualization', type=str, default='Visualization_Frequency_Convolutional_Kernels/Dataset_2b')
    parser.add_argument("--seed", type=int, default=42, help="Seed of initialization")
    parser.add_argument('--auxiliary_branch', type=str, default='True')
    parser.add_argument("--labels", type=int, default=2)
    parser.add_argument("--num_subjects", type=int, default=9)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else'cpu')
    parser.add_argument('--max_frequence', type=int, default=80)
    args = parser.parse_args()
    
    saved_path_visualization_frequency = f'{args.saved_path_visualization}/{args.name_model}'
    if not os.path.exists(saved_path_visualization_frequency):
        os.makedirs(saved_path_visualization_frequency)
    
    best_fold = find_minum_loss(f'{args.saved_path}/{args.name_model}_seed{args.seed}_validation_log.txt')
    extra_args = {'b_preds': args.auxiliary_branch} if 'MS' in args.name_model else {}
    model = (
            network_factory_methods[args.name_model](model_name_prefix=f'{args.saved_path}/{args.name_model}_seed{args.seed}',
                num_classes=args.labels, samples=args.samples, channels=args.channels, **extra_args)
        )
    model.to(args.device)
    model.load_state_dict(torch.load(f'{args.saved_path}/{args.name_model}_seed{args.seed}_best_model_fold{best_fold}.pth'))
    
    heatmap = np.zeros((36, 80))
    count=0
    for branch in range(4):
        if args.name_model == 'MSVT_SE_Net' or args.name_model == 'MSVTNet':
            conv = model.mstsconv[branch][0][0]
        else:
            conv = model.mstsconv[branch][0]
        b_matrix = conv.weight.data[:,0,0,:]
        for i in range(b_matrix.shape[0]):
            frequencies_w, frequencies_h = freqz(b_matrix[i].cpu().numpy(), fs=250, worN=125)
            heatmap[count]=np.abs(frequencies_h[:80])
            count+=1
    
    np.save('heatmap_2a_msvtnet_wout_mtl', heatmap)
    plt.figure(tight_layout=True)
    cmap=sns.color_palette("coolwarm", as_cmap=True)
    ax = sns.heatmap(heatmap.reshape(heatmap.shape[0],heatmap.shape[1]), cmap=cmap)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlim(0, heatmap.shape[1])   # asse X parte da 0
    ax.set_ylim(0, heatmap.shape[0])
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.axvline(x=0, c='black')
    plt.axvline(x=4, c='black')
    plt.axvline(x=8, c='black')
    plt.axvline(x=13, c='black')
    plt.axvline(x=30, c='black')
    # 0, 4, 8, 13, 30
    x_ticks = np.arange(0,heatmap.shape[1]+1,25).tolist() # 0 = inizio, 34 = fine
    # x_labels = [1, 5, 9, 14, 31, heatmap.shape[2]]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, rotation=45)
    
    y_ticks = [0, 9, 18, 27, 36]  # 0 = inizio, 34 = fine
    # y_labels = [1, 10, 19, 28, 37]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, rotation=90)
    
    plt.axhline(y=0, c='black')
    plt.axhline(y=9, c='black')
    plt.axhline(y=18, c='black')
    plt.axhline(y=27, c='black')
    plt.axhline(y=36, c='black')
    plt.ylabel('Feature Maps', fontsize=22)
    plt.xlabel('Frequencies (Hz)', fontsize=22)
    # plt.title('Feature Map Frequencies', fontsize=22)
    plt.tight_layout(pad=0)
    plt.savefig(f'{saved_path_visualization_frequency}/Heatmap', bbox_inches="tight", pad_inches=0.05)
    plt.close()
        # plt.savefig(f'{saved_path_visualization_frequency}/Branch_{branch+1}.png')
    # plt.savefig(f'{saved_path_visualization_frequency}/Branches.png')
    
