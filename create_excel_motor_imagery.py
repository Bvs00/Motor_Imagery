import argparse
import json
import pandas as pd
import numpy as np


# CSV Test Set
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='EEGNet')
    parser.add_argument('--path', type=str, default='Results_Augmentation')
    parser.add_argument('-full_seeds', action='store_true')
    args = parser.parse_args()

    path = args.path
    f1_score_seeds = {'Patient1':[], 'Patient2':[], 'Patient3':[], 'Patient4':[], 'Patient5':[], 'Patient6':[],
                                    'Patient7':[], 'Patient8':[], 'Patient9':[], 'Average':[]}
    dataframe = pd.DataFrame(columns=['Seed', 'Patient1', 'Patient2', 'Patient3', 'Patient4', 'Patient5', 'Patient6',
                                    'Patient7', 'Patient8', 'Patient9', 'Average'])
    dataframe_balanced = pd.DataFrame(columns=['Seed', 'Patient1', 'Patient2', 'Patient3', 'Patient4', 'Patient5', 'Patient6',
                                    'Patient7', 'Patient8', 'Patient9', 'Average'])
    dataframe_kappa = pd.DataFrame(columns=['Seed', 'Patient1', 'Patient2', 'Patient3', 'Patient4', 'Patient5', 'Patient6',
                                    'Patient7', 'Patient8', 'Patient9', 'Average'])
    list_seeds = [42, 71, 101, 113, 127, 131, 139, 149, 157, 163, 173, 181, 322, 521, 
                  402, 701, 1001, 1013, 1207, 1031, 1339, 1449, 1527, 1613, 1743, 
                  1841, 3222, 5421] if args.full_seeds else [42, 71, 101, 113, 127, 131, 
                    139, 149, 157, 163, 173, 181, 322, 521]

    for count, seed in enumerate(list_seeds):
        
        file_path = f"{path}/Final_results_{args.network}_seed{seed}.json"

        with open(file_path, 'r') as f:
            data = json.load(f)
            accuracies = [round(data[i]['Accuracy'], 3) for i in range(9)]
            balanced_accuracies = [round(data[i]['Balanced Accuracy'], 3) for i in range(9)]
            kappa = [round(data[i]['Kappa'], 3) for i in range(9)]
            
            for i in range(9):
                key = f'Patient{i+1}'
                f1_val_arr = [round(x, 3) for x in data[i]['F1 Score']]
                if f1_score_seeds[key]:
                    f1_score_seeds[key] = [a+b for a,b in zip(f1_score_seeds[key], f1_val_arr)]
                else:
                    f1_score_seeds[key] = f1_val_arr
            f1_val_arr = [round(x, 3) for x in data[9]['Average']['F1 Score']]
            if f1_score_seeds['Average']:
                f1_score_seeds['Average'] = [a+b for a,b in zip(f1_score_seeds['Average'], f1_val_arr)]
            else:
                f1_score_seeds['Average'] = f1_val_arr
            
            # Aggiungere una nuova riga al DataFrame
            dataframe.loc[count] = [seed] + accuracies + [np.mean(accuracies)]
            dataframe_balanced.loc[count] = [seed] + balanced_accuracies + [np.mean(balanced_accuracies)]
            dataframe_kappa.loc[count] = [seed] + kappa + [np.mean(kappa)]
        dataframe['Seed'] = dataframe['Seed'].astype('int32')
        dataframe_balanced['Seed'] = dataframe_balanced['Seed'].astype('int32')
        dataframe_kappa['Seed'] = dataframe_kappa['Seed'].astype('int32')
    
    tmp_acc_list, tmp_bal_acc_list, tmp_kappa_list = [], [], []
    for i in range(9):
        key = f'Patient{i+1}'
        f1_score_seeds[key] = [round(x / len(list_seeds), 3) for x in f1_score_seeds[key]]
        tmp_acc_list.append(round(dataframe[key].mean(), 3))
        tmp_bal_acc_list.append(round(dataframe_balanced[key].mean(), 3))
        tmp_kappa_list.append(round(dataframe_kappa[key].mean(), 3))
    
    f1_score_seeds['Average'] = [round(x / len(list_seeds), 3) for x in f1_score_seeds['Average']]
    dataframe.loc[len(list_seeds)] = ['Average'] + tmp_acc_list + [np.mean(tmp_acc_list)]
    dataframe_balanced.loc[len(list_seeds)] = ['Average'] + tmp_bal_acc_list + [np.mean(tmp_bal_acc_list)]
    dataframe_kappa.loc[len(list_seeds)] = ['Average'] + tmp_kappa_list + [np.mean(tmp_kappa_list)]
    
    # Salvare il DataFrame in un file CSV
    dataframe.to_excel(f'{path}/seed_results_{args.network}.xlsx', index=False)
    dataframe_balanced.to_excel(f'{path}/seed_results_{args.network}_balanced.xlsx', index=False)
    dataframe_kappa.to_excel(f'{path}/seed_results_{args.network}_kappa.xlsx', index=False)
    with open(f"{path}/seed_results_{args.network}_f1_score.json", 'w') as f:
        json.dump(f1_score_seeds, f, indent=1)
