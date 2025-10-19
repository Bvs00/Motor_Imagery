import os
import argparse
import torch
import argparse
from utils import create_tensors, find_minum_loss, validate, plot_confusion_matrix, \
    load_normalizations, available_paradigm, available_network, network_factory_methods, JointCrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, default='/mnt/datasets/eeg/Dataset_BCI_2b/Signals_BCI_3classes/test_2b_full.npz')
    parser.add_argument("--name_model", type=str, default='EEGNet', choices=available_network)
    parser.add_argument('--saved_path_mi', type=str, default='Results_4_49/Results_EEGNet/Patient')
    parser.add_argument('--saved_path_events', type=str, default='Results_4_49/Results_Events/Results_EEGNet/Patient')
    parser.add_argument('--saved_path_final', type=str, default='Results_4_49/Results_2_classifier/Results_EEGNet/Patient')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--paradigm', type=str, choices=available_paradigm)
    args = parser.parse_args()
    
    if not os.path.exists(args.saved_path_final):
        os.makedirs(args.saved_path_final)    
    
    data_test_tensors_events, labels_test_tensors_events = create_tensors(args.test_set)
    # left -> 1; right -> 2
    loss_list, f1_list, accuracy_list, balanced_accuracy_list, final_results = [], [], [], [], []
    
    if args.paradigm=='LOSO':
        dir_path, filename = os.path.split(args.test_set)
        new_filename = filename.replace("test", "train")
        train_set_path = os.path.join(dir_path, new_filename)
        data_train_tensors, labels_train_tensors = create_tensors(train_set_path)
    
    for patient in range(len(data_test_tensors_events)):
        data_events, labels_events = data_test_tensors_events[patient], labels_test_tensors_events[patient]
        
        saved_path_events = args.saved_path_events if args.paradigm=='Cross' else f'{args.saved_path_events}/Patient_{patient + 1}'
        mean_data_events, std_data_events, min_data_events, max_data_events = load_normalizations(f'{saved_path_events}/{args.name_model}')
        
        if mean_data_events != None:
            data_events = (data_events - mean_data_events)/std_data_events
        else:
            data_events = (data_events - min_data_events)/(max_data_events - min_data_events)
        
        dataset_events = TensorDataset(data_events, labels_events)
        test_events_loader = DataLoader(dataset_events, batch_size=256, num_workers=5)

        best_fold_events = find_minum_loss(f'{saved_path_events}/{args.name_model}_seed{args.seed}_validation_log.txt')

        model_events = (
            network_factory_methods[args.name_model](model_name_prefix=f'{saved_path_events}/{args.name_model}_seed{args.seed}',
                 num_classes=2,
                 samples=data_events.shape[3], channels=data_events.shape[2])
        )
        model_events.to(args.device)
        model_events.load_state_dict(torch.load(f'{saved_path_events}/{args.name_model}_seed{args.seed}_best_model_fold{best_fold_events}.pth'))

        # Motor Imagery
        saved_path_mi = args.saved_path_mi if args.paradigm=='Cross' else f'{args.saved_path_mi}/Patient_{patient + 1}'
        mean_data, std_data, min_data, max_data = load_normalizations(f'{saved_path_mi}/{args.name_model}')
        
        best_fold = find_minum_loss(f'{saved_path_mi}/{args.name_model}_seed{args.seed}_validation_log.txt')
        model = (
            network_factory_methods[args.name_model](model_name_prefix=f'{saved_path_mi}/{args.name_model}_seed{args.seed}',
                 num_classes=2,
                 samples=data_events.shape[3], channels=data_events.shape[2])
        )
        model.to(args.device)
        model.load_state_dict(torch.load(f'{saved_path_mi}/{args.name_model}_seed{args.seed}_best_model_fold{best_fold}.pth'))

        model_events.eval()
        model.eval()
        
        all_preds = []
        all_labels = []
        mean_data_events = mean_data_events.to(args.device)
        std_data_events = std_data_events.to(args.device)
        mean_data = mean_data.to(args.device)
        std_data = std_data.to(args.device)
        labels = []
        with torch.no_grad():
            for raw, label in test_events_loader:
                raw, label = raw.to(args.device), label.to(args.device)
                labels.extend(label.cpu().numpy())
                outputs_events = model_events(raw)
                _, preds_events = torch.max(outputs_events, 1) # first prediction
                
                raw_filtered = raw[preds_events == 1]
                label_filtered = label[preds_events == 1]
                
                # return in original distribution
                if mean_data_events != None:
                    raw_filtered = (raw_filtered * std_data_events) + mean_data_events
                else:
                    raw_filtered = (raw_filtered * (max_data_events - min_data_events)) + min_data_events
                
                # raw_filtered = (raw_filtered-mean_data)/std_data # normalize in according to second classifier
                if mean_data != None:
                    raw_filtered = (raw_filtered - mean_data)/std_data
                else:
                    raw_filtered = (raw_filtered - min_data)/(max_data - min_data)
                
                outputs = model(raw_filtered)
                _, preds = torch.max(outputs, 1) # second prediction
                
                final_prediction = torch.clone(preds_events)
                final_prediction[final_prediction==1] = preds+1
                
                all_preds.extend(final_prediction.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

            f1 = f1_score(all_labels, all_preds, average=None).tolist()
            accuracy = accuracy_score(all_labels, all_preds)
            balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
            conf_matrix = confusion_matrix(all_labels, all_preds)

        saved_path_final = args.saved_path_final if args.paradigm=='Cross' else f'{args.saved_path_final}/Patient_{patient + 1}'
        
        if not os.path.exists(saved_path_final):
            os.makedirs(saved_path_final)
            
        plot_confusion_matrix(conf_matrix, ['Background', 'Left Hand', 'Right Hand'] if conf_matrix.shape[0]==3 else ['Left Hand', 'Right Hand'], best_fold,
                              f'{saved_path_final}/{args.name_model}_seed{args.seed}_test', balanced_accuracy)
        final_results.append({'Patient': patient+1, 'F1 Score': f1, 'Accuracy': accuracy, 'Balanced Accuracy': balanced_accuracy})
        f1_list.append(f1), accuracy_list.append(accuracy), balanced_accuracy_list.append(balanced_accuracy)
        
    final_results.append(
        {f"Average": {"F1 Score": np.mean(f1_list, axis=0).tolist(), "Accuracy": np.mean(accuracy_list), "Balanced Accuracy": np.mean(balanced_accuracy_list)}}
    )
    with open(f'{args.saved_path_final}/Final_results_{args.name_model}_seed{args.seed}.json', 'w') as f:
        json.dump(final_results, f, indent=1)

    print('All patient test results have been saved.')
