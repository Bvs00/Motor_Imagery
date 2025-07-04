import random
import numpy as np
import torch


def chr_augmentation(data, labels):
    """
    Preso il dato di input [batch, 1, 3, 1000] devo invertire i canali e invertire la labels.
    
    """
    data_inverted = torch.zeros_like(data)
    data_inverted[:, :, [0, 2], :] = data[:, :, [2, 0], :]
    
    num_segments=8
    length_segment=data.shape[3]//num_segments
    
    new_data = torch.empty_like(data)
    
    for i in range(data.shape[0]):
        for seg_idx in range(num_segments):
            # Selezioniamo casualmente un campione dal batch originale
            random_sample = random.randint(0, data.shape[0] - 1)
            
            # Prendiamo il segmento corretto e lo assembliamo nel nuovo campione
            start = seg_idx * length_segment
            end = (seg_idx + 1) * length_segment
            
            # Assembliamo mantenendo la struttura [batch, 1, 3, 1000]
            new_data[i, :, :, start:end] = data[random_sample, :, :, start:end]

    return torch.cat([data, data_inverted, new_data]), torch.cat([labels, (1-labels), labels])

def reverse_channels(data,labels):
    """
    Preso il dato di input [batch, 1, 3, 1000] devo invertire i canali e invertire la labels.
    
    """
    data_inverted = torch.zeros_like(data)
    data_inverted[:, :, [0, 2], :] = data[:, :, [2, 0], :]
    
    return torch.cat([data, data_inverted]), torch.cat([labels, (1-labels)])

def segmentation_reconstruction(data, labels, num_segments=8):
    full_data = data.clone()
    full_labels = labels.clone()
    type_labels = torch.unique(labels)
    length_segment=data.shape[3]//num_segments
    for label in type_labels:
        idx_labels=torch.where(labels == label)[0]
        data_for_label = data[idx_labels,:,:,:]
        labels_tmp = labels[idx_labels]
        new_data = torch.zeros_like(data_for_label)
        for i in range(data_for_label.shape[0]):
            for seg_idx in range(num_segments):
                # Selezioniamo casualmente un campione dal batch originale
                random_sample = random.randint(0, data_for_label.shape[0] - 1)
                # Prendiamo il segmento corretto e lo assembliamo nel nuovo campione
                start = seg_idx * length_segment
                end = (seg_idx + 1) * length_segment
                # Assembliamo mantenendo la struttura [batch, 1, 3, 1000]
                new_data[i, :, :, start:end] = data_for_label[random_sample, :, :, start:end]
        full_data = torch.cat([full_data, new_data], dim=0)
        full_labels = torch.cat([full_labels, labels_tmp], dim=0)
    
    idx_shuffled = torch.randperm(full_data.size(0))
    return full_data[idx_shuffled], full_labels[idx_shuffled]

    
    