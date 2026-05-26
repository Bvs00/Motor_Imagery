#!/bin/bash

if [ -z "$PRIME" ] || [ -z "$DATASET" ] || [ -z "$AUX" ]; then
    echo "Errore: Devi specificare PRIME, DATASET, AUG!"
    exit 1
fi

echo "$PRIME"
echo "$DATASET"
echo "$AUX"

networks=('EEGNet' 'EEGConformer' 'CTNet' 'MSVTNet')

if [ "$PRIME" == "1" ]; then
  primes=(42 71 101 113 127 131 139)
elif [ "$PRIME" == "2" ]; then
  primes=(149 157 163 173 181 322 521)
fi

echo "${primes[@]}"
echo "${networks[@]}"
aug="segmentation_reconstruction"
normalization="Z_Score_unique"
bandpass="no_full"
paradigm="Cross"
classes="2"
dataset="$DATASET"
aux="$AUX"
saved_path="No_Full/Results_${dataset}/Results_Z_Score_unique/Results_SegRec/Results_Cross"
echo "$saved_path"

# /home/inbit/Scrivania/Datasets/2B/
# /mnt/datasets/eeg/Dataset_BCI_2b/Signals_BCI_2classes/

for network in "${networks[@]}"; do
    echo "$network"
    for seed in "${primes[@]}"; do
    echo "Train seed: $seed"
    python -u train_motor_imagery.py --seed "$seed" --name_model "$network" --saved_path "${saved_path}/Results_${network}" --lr 0.001 \
            --augmentation "$aug" --num_workers 5 --normalization "$normalization" --paradigm "$paradigm" \
            --train_set "/cache/sbove/datasets/eeg/Dataset_BCI_${dataset}/Signals_BCI_${classes}classes/train_${dataset}_$bandpass.npz" \
            --patience 150 --batch_size 72 --auxiliary_branch "$aux"
    python -u test_motor_imagery.py --name_model "$network" --saved_path "${saved_path}/Results_${network}" --paradigm "$paradigm" \
            --test_set "/cache/sbove/datasets/eeg/Dataset_BCI_${dataset}/Signals_BCI_${classes}classes/test_${dataset}_$bandpass.npz" \
            --seed "$seed" --auxiliary_branch "$aux"
    done
    python create_excel_motor_imagery.py --network "$network" --path "${saved_path}/Results_${network}"
    echo 'ok'
done

