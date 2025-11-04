from utils import create_tensors, normalization_z_score_unique
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA

# train_set_path='/mnt/datasets/eeg/Dataset_BCI_2b/Signals_BCI_2classes/train_2b_full.npz'
train_set_path = '/mnt/datasets/eeg/Dataset_BCI_2a/train_2a_4_40.npz'
data_train_tensors, labels_train_tensors = create_tensors(train_set_path)

num_subjects = len(data_train_tensors)
cmap = mpl.colormaps.get_cmap('tab10').resampled(num_subjects)
plt.figure(figsize=(10, 8))

for subject in range(len(data_train_tensors)):
    print(subject+1)
    data = data_train_tensors[subject]
    # mean, std, _, _ = normalization_z_score_unique(data)
    # data = (data - mean)/std
    data = torch.reshape(data, [data.shape[0], -1]).numpy()
    data_embedded_pca = PCA(n_components=50).fit_transform(data)
    data_embedded = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(data_embedded_pca)
    # breakpoint()
    plt.scatter(
        data_embedded[:, 0],
        data_embedded[:, 1],
        s=20,
        color=cmap(subject / num_subjects),
        alpha=0.6,
        label=f"Subject {subject+1}"
    )

plt.title("t-SNE Embedding of Subjects of Dataset BCI IV 2b")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.tight_layout()
plt.savefig('2D_tSNE_BCI_IV_2a')