import numpy as np
import mne
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt
import seaborn as sns



def plot_spect(spect, cmap, title, plt_title):
    plt.figure(figsize=(10,6))
    plt.imshow(spect, aspect="auto", origin="lower", cmap=cmap, extent=[0, 4, 8, 30])
    # plt.imshow(spect, aspect="auto", origin="lower", cmap=cmap)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Time (s)', fontsize=26)
    plt.ylabel('Frequency (Hz)', fontsize=26)
    plt.title(plt_title, fontsize=30)
    cbar = plt.colorbar()
    cbar.set_label("Normalized Power", fontsize=26)
    cbar.ax.tick_params(labelsize=26)
    plt.tight_layout()
    plt.savefig(title)



# cmap=sns.color_palette("coolwarm", as_cmap=True)
# cmap=sns.color_palette("Spectral", as_cmap=True)
cmap=sns.color_palette("viridis", as_cmap=True)
cmap.set_bad('black')

dataset = np.load('/home/inbit/Scrivania/Datasets/2b/Signals_BCI_2classes/train_2b_full.npz', allow_pickle=True)
# dataset = np.load('/home/inbit/Scrivania/Datasets/2b/Signals_BCI_3classes/train_2b_full.npz', allow_pickle=True)
data = dataset['data'][0]
label = dataset['labels'][0]

N, C, _ = data.shape

data = (data-np.mean(data, axis=(None), keepdims=True))/np.std(data, axis=(None), keepdims=True)

#filtraggio
b, a = butter(5, [8.0, 30.0], btype='bandpass', fs=250)

data_filt = np.zeros_like(data)

for i in range(N):
    for j in range(C):
        data_filt[i, j] = filtfilt(b, a, data[i, j])
        


cwt_coefficients = mne.time_frequency.tfr_array_morlet(data_filt, sfreq=250, freqs=np.linspace(8, 30, 32), n_jobs=10, output='power', n_cycles=7)
# cwt_coefficients = mne.time_frequency.tfr_array_morlet(data, sfreq=250, freqs=np.linspace(4, 80, 96), n_jobs=10, output='power', n_cycles=7)


cwt_coefficients = np.log1p(cwt_coefficients)
    
mean = np.mean(cwt_coefficients, axis=(0), keepdims=True)
std = np.std(cwt_coefficients, axis=(0), keepdims=True)

cwt_coefficients = (cwt_coefficients - mean) / (std + 1e-10)

original_scalogram = cwt_coefficients[0, 0]
H, W = original_scalogram.shape

# original_scalogram = original_scalogram/np.max(original_scalogram, axis=(None), keepdims=True)
plot_spect(original_scalogram, cmap, 'zzz', 'Original')

dim_h = H//8
dim_w = W//8

original_patches = original_scalogram.reshape(dim_h, 8, dim_w, 8)
original_patches = original_patches.transpose(0,2,1,3)
original_patches = original_patches.reshape(-1, 8, 8)

patches = np.copy(original_patches)

pos_ini = 125
pos_fin = 225

patches[pos_ini:pos_fin, :, :] = np.nan
cmap2 = sns.color_palette("viridis", as_cmap=True)
cmap2.set_bad('black')

easy_spectrogram = patches.reshape(dim_h, dim_w, 8, 8)
easy_spectrogram = easy_spectrogram.transpose(0, 2, 1, 3)
easy_spectrogram = easy_spectrogram.reshape(easy_spectrogram.shape[0]*easy_spectrogram.shape[1], (easy_spectrogram.shape[2]*easy_spectrogram.shape[3]))

plot_spect(easy_spectrogram, cmap, 'zzz_easy', 'Easy Step')

space=3
random_pos = sorted(np.random.randint(0, 1000, (100)))

patches = np.copy(original_patches)
for pos in random_pos:
    patches[pos:pos+space, :, :] = np.nan
    
medium_spectrogram = patches.reshape(dim_h, dim_w, 8, 8)
medium_spectrogram = medium_spectrogram.transpose(0, 2, 1, 3)
medium_spectrogram = medium_spectrogram.reshape(medium_spectrogram.shape[0]*medium_spectrogram.shape[1], (medium_spectrogram.shape[2]*medium_spectrogram.shape[3]))

plot_spect(medium_spectrogram, cmap, 'zzz_medium', 'Medium Step')

space=3
random_pos = sorted(np.random.randint(0, 1000, (300)))

patches = np.copy(original_patches)
for pos in random_pos:
    patches[pos:pos+space, :, :] = np.nan
    
hard_spectrogram = patches.reshape(dim_h, dim_w, 8, 8)
hard_spectrogram = hard_spectrogram.transpose(0, 2, 1, 3)
hard_spectrogram = hard_spectrogram.reshape(hard_spectrogram.shape[0]*hard_spectrogram.shape[1], (hard_spectrogram.shape[2]*hard_spectrogram.shape[3]))

plot_spect(hard_spectrogram, cmap, 'zzz_hard', 'Hard Step')


# from_scratch = [71.94, 57.21, 55.28, 91.08, 81.22, 78.89, 68.89, 76.84, 81.25, 73.62]
# std_1 = 11.01
# pretrained = [77.36, 61.03, 57.22, 92.30, 76.76, 82.50, 70.83, 79.47, 79.86, 75.26]
# std_2 = 10.20

# xlabel = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09']
# xlabel = np.arange(9)

# colors = sns.color_palette()
# width=0.25
# plt.figure(figsize=(10,6))
# offset = (-9/2) * width + width/2
# plt.bar(xlabel+offset, from_scratch[:-1], width, label='From Scratch', color=colors[0])
# # plt.bar(xlabel[-1]+offset, from_scratch[-1], width, yerr=std_1, capsize=6, color=colors[0])
# offset = (1-9/2) * width + width/2
# plt.bar(xlabel+offset, pretrained[:-1], width, label='CSSL Pretrained', color=colors[1])
# # plt.bar(xlabel[-1]+offset, pretrained[-1], width, yerr=std_2, capsize=6, color=colors[1])
# plt.xticks(xlabel-0.885, ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09'), fontsize=20)
# plt.yticks(fontsize=22)
# plt.tick_params(axis='y', labelsize=20)
# plt.legend(loc='upper left', ncols=3, fontsize=22)
# plt.ylim(top=110)
# plt.title('', fontsize=20)
# plt.xlabel('Subject', fontsize=22)
# plt.ylabel('Accuracy (%)', fontsize=22)
# plt.tight_layout(pad=0)
# plt.savefig('zzz_bar', bbox_inches="tight")

