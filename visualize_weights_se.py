import sys
import os
from utils import create_tensors, find_minum_loss, validate, \
    load_normalizations, available_paradigm, available_network, network_factory_methods, JointCrossEntropyLoss
import argparse
from torch.utils.data import TensorDataset, DataLoader
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def validate_visualization_senet(model, val_loader, criterion_tasks, device, return_attention=True):
    model.eval()
    all_preds_tasks = []
    all_labels_tasks = []
    all_se_weights_branches = {}
    all_se_weights = []
    with torch.no_grad():
        for x_raw_batch, labels_batch in val_loader:
            x_raw_batch, labels_batch = x_raw_batch.to(device), labels_batch.to(device)
            if args.auxiliary_branch:
                output_tasks, _, se_weights_branches, se_weights = model(x_raw_batch, return_attention)
            else:
                output_tasks, se_weights_branches, se_weights = model(x_raw_batch, return_attention)
            if se_weights_branches is not None:
                for count, se_weights_branch in enumerate(se_weights_branches):
                    if count not in all_se_weights_branches:
                        all_se_weights_branches[count]=[]
                        all_se_weights_branches[count].extend(se_weights_branch.detach().cpu().numpy())
                    else:
                        all_se_weights_branches[count].extend(se_weights_branch.detach().cpu().numpy())
                    
            if se_weights is not None:
                all_se_weights.extend(se_weights.detach().cpu().numpy())
            
            _, preds_tasks = torch.max(output_tasks, 1)
            all_preds_tasks.extend(preds_tasks.cpu().numpy())
            all_labels_tasks.extend(labels_batch.cpu().numpy())
        
        if len(all_se_weights_branches.keys()) != 0:
            for key in all_se_weights_branches:
                all_se_weights_branches[key]=np.squeeze(np.stack(all_se_weights_branches[key]), axis=(2,3))
        
        if len(all_se_weights) != 0:
            all_se_weights = np.squeeze(np.stack(all_se_weights), axis=(2,3))
        balanced_accuracy_tasks = balanced_accuracy_score(all_labels_tasks, all_preds_tasks)

    return balanced_accuracy_tasks, all_se_weights_branches, all_se_weights


def plot_se_weights(se_weights, subject, most_important, threshold, name_path):
    path = f'{args.saved_path_plot}/{args.name_model}'
    if not os.path.exists(path):
        os.makedirs(path)
    
    mean_se_weights = np.mean(se_weights, axis=0)
    std_se_weights = np.std(se_weights, axis=0)
    plt.figure(figsize=(18,10))
    plt.stem(mean_se_weights, label='Mean of Value')
    plt.errorbar(
        range(len(mean_se_weights)),
        mean_se_weights,
        yerr=std_se_weights,
        fmt='none',          # no marker, no connecting line
        ecolor='red',        # color of the error bars
        elinewidth=1.5,
        capsize=3,
        label='Std Dev'
    )
    plt.title("SE Attention Weights per Channel")
    plt.xlabel("Channel")
    plt.ylabel("Attention weight")
    plt.legend()
    plt.savefig(f'{path}/{name_path}_{subject+1}.png')
    plt.close()
    most_important.append({f'Subject {subject+1}': np.where(mean_se_weights > threshold)[0].tolist()})
    return mean_se_weights

def plot_heatmap(matrix_se_weights, path='prova.png'):
    h, w = matrix_se_weights.shape
    plt.figure(figsize=(w * 0.6, h * 0.6))
    sns.heatmap(matrix_se_weights, annot=True, fmt='.2f', vmin=0, vmax=1, annot_kws={"size": 10}, cmap="coolwarm") # sulle ascisse c'è la prima dimensione mentre sulle ordinate la seconda dimensione
    plt.ylabel('Subjects')
    plt.xlabel('Features')
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, default='/mnt/datasets/eeg/Dataset_BCI_2b/Signals_BCI_2classes/test_2b_full.npz')
    parser.add_argument("--name_model", type=str, default='MSVT_SE_SE_Net', help="Name of model that use", choices=available_network)
    parser.add_argument('--saved_path', type=str, default='Results_Black/Results_Z_Score_unique/Results_SegRec/Results_Cross/Results_MSVT_SE_SE_Net_Full_NoBandpass')
    parser.add_argument('--saved_path_plot', type=str, default='Visualization_SE_Weights')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--paradigm', type=str, choices=available_paradigm, default='Cross')
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--auxiliary_branch', type=str, default='False')
    args = parser.parse_args()
    
    args.auxiliary_branch = True if args.auxiliary_branch == 'True' else False
    
    data_test_tensors, labels_test_tensors = create_tensors(args.test_set)
    num_subjects=9
    most_important_branches = {}
    most_important = []
    
    matrix_se_weights_inside = np.zeros((4, 9, 9)) # 4 branches 9 soggetti e 9 feature maps
    matrix_se_weights_outside = np.zeros((9, 72)) # 9 soggetti e 72 feature maps
    
    for patient in range(len(data_test_tensors)):
        data, labels = data_test_tensors[patient], labels_test_tensors[patient]
        
        saved_path = args.saved_path if args.paradigm=='Cross' else f'{args.saved_path}/Patient_{patient + 1}'
        
        mean, std, min_, max_ = load_normalizations(f'{saved_path}/{args.name_model}')
        
        
        if mean != None:
            data = (data - mean)/std
        else:
            data = (data - min_)/(max_ - min_)

        dataset = TensorDataset(data, labels)
        test_loader = DataLoader(dataset, batch_size=256, num_workers=5)

        best_fold = find_minum_loss(f'{saved_path}/{args.name_model}_seed{args.seed}_validation_log.txt')
        extra_args = {'b_preds': args.auxiliary_branch} if 'MS' in args.name_model else {}
        model = (
            network_factory_methods[args.name_model](model_name_prefix=f'{saved_path}/{args.name_model}_seed{args.seed}',
                num_classes=len(np.unique(labels)),
                samples=data.shape[3], channels=data.shape[2], **extra_args)
        )
        model.to(args.device)

        old_state = torch.load(f'{saved_path}/{args.name_model}_seed{args.seed}_best_model_fold{best_fold}.pth')
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

        if (args.name_model == "MSVTNet" or args.name_model == "MSVTSENet" or args.name_model == "MSVT_SE_Net" or args.name_model == "MSVT_SE_SE_Net") and (args.auxiliary_branch):
            criterion_tasks = JointCrossEntropyLoss()
        else:
            criterion_tasks = nn.CrossEntropyLoss()
        
        bacc_task, se_weights_branches, se_weights = validate_visualization_senet(model, test_loader, criterion_tasks, device=args.device, return_attention=True)

        if len(se_weights_branches.keys()) != 0:
            for count in se_weights_branches:
                if count not in most_important_branches:
                    most_important_branches[count]=[]
                matrix_se_weights_inside[count][patient] = plot_se_weights(se_weights_branches[count], patient, most_important_branches[count], args.threshold, f'se_weights_branch_{count}')

        if len(se_weights) != 0:
            matrix_se_weights_outside[patient]=plot_se_weights(se_weights, patient, most_important, args.threshold, 'se_weights')

    if len(most_important_branches)!=0:
        with open(f'{args.saved_path_plot}/{args.name_model}/Most_Important_SE_Weights_Inside_Branches.json', 'w') as f:
                    json.dump(most_important_branches, f, indent=4)
                
    if len(most_important)!=0:
        with open(f'{args.saved_path_plot}/{args.name_model}/Most_Important_SE_Weights_Outside_Branches.json', 'w') as f:
            json.dump(most_important, f, indent=4)
    # print most important features among subjects
    count_feature_maps = {}
    for sub, elements in enumerate(most_important):
        values = elements[f'Subject {sub+1}']
        for value in values:
            if value not in count_feature_maps:
                count_feature_maps[value]=1
            else: 
                count_feature_maps[value]+=1
    sorted_dict = dict(sorted(count_feature_maps.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)
    if matrix_se_weights_inside.any() != 0:
        branches, subjects, features = matrix_se_weights_inside.shape
        matrix_se_weights_inside = matrix_se_weights_inside.transpose(1, 0, 2).reshape(subjects, branches*features)
        plot_heatmap(matrix_se_weights_inside, path=f'{args.saved_path_plot}/{args.name_model}/Heatmap_inside_matrix.png')
    if matrix_se_weights_outside.any() != 0:
        plot_heatmap(matrix_se_weights_outside, path=f'{args.saved_path_plot}/{args.name_model}/Heatmap_outside_matrix.png')
