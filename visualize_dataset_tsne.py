from utils import create_tensors_subjects, load_normalizations, find_minum_loss,\
    available_network, available_paradigm
from Network_Visualization import network_factory_methods
import argparse
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns


def plot_data(data_embedded, y, colors, markers, path, name_file, title):
    if not os.path.exists(f'{args.saved_path_plot}/{path}'):
        os.makedirs(f'{args.saved_path_plot}/{path}')
    
    class_labels = {
        0: 'Left Hand',
        1: 'Right Hand'
    }
    
    plt.figure(figsize=(9, 7), dpi=300)
    
    plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "legend.fontsize": 13
    })
    
    classes = np.unique(y)
    flag_subjects = False
    if len(classes) >=4:
        flag_subjects=True
    for i, cls in enumerate(classes):
        idx = y == cls
        plt.scatter(
            data_embedded[idx, 0],
            data_embedded[idx, 1],
            s=30,
            color=colors[i],
            marker=markers[i],
            alpha=0.7,
            edgecolor='black',    # bordo nero
            linewidth=0.4,
            label=f"Subject {cls+1}" if flag_subjects else f"{class_labels[cls]}"
        )
    
    plt.xticks([])  
    plt.yticks([])
    # plt.xlabel("t-SNE 1")
    # plt.ylabel("t-SNE 2")
    # plt.title(title)

    # Legenda fuori dal grafico
    plt.legend(
        title="Legend",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()
    plt.savefig(f'{args.saved_path_plot}/{path}/{name_file}.png', bbox_inches="tight", pad_inches=0.05)
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=str,
                        default="/mnt/datasets/eeg/Dataset_BCI_2b/Signals_BCI_2classes/test_2b_full.npz", help="Path to train set file")
    parser.add_argument("--name_model", type=str, default='MSVTNet', help="Name of model that use", choices=available_network)
    parser.add_argument('--saved_path', type=str, default='Results_Black/Results_Z_Score_unique/Results_SegRec/Results_Cross/Results_MSVTNet_Full_NoBandpass')
    parser.add_argument('--saved_path_plot', type=str, default='Visualization_TSNE')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--paradigm', type=str, choices=available_paradigm, default='Cross')
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--auxiliary_branch', type=str, default='True')
    parser.add_argument('--feature_maps', nargs='+', type=int, default=[9, 9, 9, 9])
    parser.add_argument('--p1', type=int, default=8)
    parser.add_argument('--p2', type=int, default=7)
    args = parser.parse_args()
    
    args.auxiliary_branch = True if args.auxiliary_branch == 'True' else False
    
    data_train_tensors, labels_train_tensors, labels_train_subjects = create_tensors_subjects(args.train_set)
    mean, std, min_, max_ = load_normalizations(f'{args.saved_path}/{args.name_model}')
    data = torch.cat(data_train_tensors)
    labels = torch.cat(labels_train_tensors)
    labels_subjects = torch.cat(labels_train_subjects)
    
    # colors = [
    # "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    # "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"
    # ]
    colors = sns.color_palette("bright", n_colors=9)
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>']
    
    if mean != None:
        data = (data - mean)/std
    else:
        data = (data - min_)/(max_ - min_)
    
    data_reshape = torch.reshape(data, [data.shape[0], -1]).numpy()
    data_embedded_pca = PCA(n_components=50).fit_transform(data_reshape)
    data_embedded = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(data_embedded_pca)
    y_subjects = labels_subjects.numpy()
    y_tasks = labels.numpy()
    plot_data(data_embedded, y=y_subjects, colors=colors, markers=markers, path='Dataset_2B', name_file='Original_Data_Subjects', \
        title='2D Representation of Original Data grouped for Subjects')
    plot_data(data_embedded, y=y_tasks, colors=colors, markers=markers, path='Dataset_2B', name_file='Original_Data_Tasks', \
        title='2D Representation of Original Data grouped for Tasks')
    
    best_fold = find_minum_loss(f'{args.saved_path}/{args.name_model}_seed{args.seed}_validation_log.txt')
    extra_args = {'b_preds': args.auxiliary_branch, 'F': args.feature_maps, 'P1': args.p1, 'P2': args.p2} if 'MS' in args.name_model else {}
    model = (
        network_factory_methods[args.name_model](model_name_prefix=f'{args.saved_path}/{args.name_model}_seed{args.seed}',
            num_classes=len(np.unique(labels)),
            samples=data.shape[3], channels=data.shape[2], **extra_args)
    )
    model.to(args.device)
    
    old_state = torch.load(f'{args.saved_path}/{args.name_model}_seed{args.seed}_best_model_fold{best_fold}.pth')
    
    if args.name_model == 'MSVT_SE_SE_Net':
            
        new_state = {}

        # mapping table
        map_table = {
            "0": "conv1",
            "1.fc1": "se1.fc1",
            "1.fc2": "se1.fc2",
            "2": "bn1",
            "3": "depth_conv1",
            "4": "bn2",
            "8": "depth_conv2",
            "9": "bn3",
        }

        for k, v in old_state.items():

            if not k.startswith("mstsconv."):
                new_state[k] = v
                continue

            parts = k.split(".")
            branch = parts[1]           # "0","1","2","3"
            rest = ".".join(parts[2:])  # e.g., "1.fc1.weight" or "0.weight"

            prefix = rest.split(".")[0]  # e.g. "1", "2", "8"

            # Special case for fc1/fc2
            if prefix == "1" and "fc1" in rest:
                lookup = "1.fc1"
                tail = rest.split("fc1.")[1]  # "weight"
            elif prefix == "1" and "fc2" in rest:
                lookup = "1.fc2"
                tail = rest.split("fc2.")[1]
            else:
                lookup = prefix
                tail = rest[len(prefix)+1:] if "." in rest else ""

            if lookup in map_table:
                new_key = f"mstsconv.{branch}.{map_table[lookup]}"
                if tail:
                    new_key += f".{tail}"
                new_state[new_key] = v
            else:
                print("Skipping:", k)
    elif args.name_model == 'MSVT_SE_Net': 
        mapping = {
            "0": "conv1",
            "1": "se1",
            "2": "bn1",
            "3": "depth_conv1",
            "4": "bn2",
            "8": "depth_conv2",
            "9": "bn3",
        }

        new_state = {}

        for k, v in old_state.items():

            # se non è un TSConv, COPIALA COSÌ COM'È
            if not k.startswith("mstsconv."):
                new_state[k] = v
                continue

            # parsing chiavi TSConv vecchio modello
            parts = k.split(".")
            # esempio: mstsconv.0.0.2.weight → ["mstsconv","0","0","2","weight"]

            block = parts[1]     # es: "0", "1", "2", "3"
            idx = parts[3]       # 0,1,2,3,4,8,9

            # se il layer non ha pesi, lo ignoriamo
            if idx not in mapping:
                continue

            new_layer = mapping[idx]
            tail = ".".join(parts[4:])  # weight, bias, running_mean, ecc.

            # AGGIUNGIAMO IL LIVELLO ".0." MANCANTE
            new_key = f"mstsconv.{block}.0.{new_layer}.{tail}"

            new_state[new_key] = v
    else: 
        new_state=old_state
    
    model.load_state_dict(new_state)
    data = data.to(args.device)
    
    with torch.no_grad():
        model.eval()
        _, _, latent_representation = model(data)
    
    latent_representation = latent_representation.cpu().numpy()
    latent_representation_pca = PCA(n_components=50).fit_transform(latent_representation)
    latent_representation_embedded = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(latent_representation_pca)
    plot_data(latent_representation_embedded, y=y_subjects, colors=colors, markers=markers, path='Dataset_2B', \
        name_file='MSVTNet_Data_Subjects', title='2D Representation of Latent Representation of Original Data grouped for Subjects')
    plot_data(latent_representation_embedded, y=y_tasks, colors=colors, markers=markers, path='Dataset_2B', \
        name_file='MSVTNet_Data_Tasks', title='2D Representation of Latent Representation of Original Data grouped for Tasks')
    print('finish')
        
    