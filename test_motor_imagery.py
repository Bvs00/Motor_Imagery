import sys
from utils import create_tensors, find_minum_loss, validate, plot_confusion_matrix, \
    load_normalizations, available_paradigm, available_network, network_factory_methods
import argparse
from torch.utils.data import TensorDataset, DataLoader
import json
import numpy as np
import torch
import torch.nn as nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, default='/mnt/datasets/eeg/Dataset_BCI_2b/Signals_BCI_2classes/test_2b_full.npz')
    parser.add_argument("--name_model", type=str, default='EEGConformer', help="Name of model that use", choices=available_network)
    parser.add_argument('--saved_path', type=str, default='Results_EEGConformer')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--paradigm', type=str, choices=available_paradigm)
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

        dataset = TensorDataset(data, labels)
        test_loader = DataLoader(dataset, batch_size=256, num_workers=5)

        best_fold = find_minum_loss(f'{saved_path}/{args.name_model}_seed{args.seed}_validation_log.txt')

        model = (
            network_factory_methods[args.name_model](model_name_prefix=f'{saved_path}/{args.name_model}_seed{args.seed}',
                num_classes=len(np.unique(labels)),
                samples=data.shape[3], channels=data.shape[2])
        )
        model.to(args.device)
        model.load_state_dict(torch.load(f'{saved_path}/{args.name_model}_seed{args.seed}_best_model_fold{best_fold}.pth'))

        criterion = nn.CrossEntropyLoss()

        avg_loss, f1, confusion_matrix, accuracy, balanced_accuracy = validate(model, test_loader, criterion, args.device)

        plot_confusion_matrix(confusion_matrix, ['Background', 'Left Hand', 'Right Hand'] if confusion_matrix.shape[0]==3 else ['Left Hand', 'Right Hand'], best_fold,
                            f'{saved_path}/{args.name_model}_seed{args.seed}_test', balanced_accuracy)

        if args.paradigm == 'Single' or args.paradigm == 'LOSO':
            with open(f'{saved_path}/{args.name_model}_seed{args.seed}_test_results.json', 'w') as f:
                json.dump({'average_loss': avg_loss, 'f1_score': f1, 'accuracy': accuracy, 'balanced_accuracy': balanced_accuracy}, f, indent=4)

        loss_list.append(avg_loss), f1_list.append(f1), accuracy_list.append(accuracy), balanced_accuracy_list.append(balanced_accuracy)

        final_results.append({'Patient': patient+1, 'Loss': avg_loss, 'F1 Score': f1, 'Accuracy': accuracy, 'Balanced Accuracy': balanced_accuracy})

    final_results.append(
        {f"Average": {"Loss": np.mean(loss_list), "F1 Score": np.mean(f1_list, axis=0).tolist(), "Accuracy": np.mean(accuracy_list), "Balanced Accuracy": np.mean(balanced_accuracy_list)}}
    )

    with open(f'{args.saved_path}/Final_results_{args.name_model}_seed{args.seed}.json', 'w') as f:
        json.dump(final_results, f, indent=4)

    print('All patient test results have been saved.')
    
    sys.exit()
