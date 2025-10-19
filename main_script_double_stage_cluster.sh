#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --ntasks=1
#SBATCH --account=lm_foggia
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=07:00:00
#SBATCH --nodelist=gnode11
#SBATCH --job-name=EEGNet_single_double_1
#SBATCH --output=EEGNet_single_double_1.log


export TORCH_DEVICE=cuda
export PYTHON=/home/bvosmn000/.conda/envs/ICareMeEnv/bin/python

if [ -z "$NET" ] || [ -z "$PRIME" ] || [ -z "$AUG" ] || [ -z "$SAVED_PATH_EVENTS" ] || [ -z "$SAVED_PATH_MI" ] || [ -z "$SAVED_PATH_FINAL" ] || [ -z "$NORM" ] || [ -z "$BANDPASS" ] || [ -z "$PARADIGM" ]; then
    echo "Errore: Devi specificare NET, PRIME, AUG, SAVED_PATH, NORM, BANDPASS, PARADIGM!"
    echo "Utilizzo: NET=<valore> PRIME=<valore> AUG=<valore> ./script.sh"
    exit 1
fi

echo "$NET"
echo "$PRIME"
echo "$AUG"
echo "$SAVED_PATH_EVENTS"
echo "$SAVED_PATH_MI"
echo "$SAVED_PATH_FINAL"
echo "$NORM"
echo "$BANDPASS"
echo "$PARADIGM"


if [ "$PRIME" == "1" ]; then
  primes=(42 71 101 113 127 131 139 149 157 163 173 181 322 521)
elif [ "$PRIME" == "2" ]; then
  primes=(402 701 1001 1013 1207 1031 1339 1449 1527 1613 1743 1841 3222 5421)
elif [ "$PRIME" == "3" ]; then
  primes=(42 71 101 113 127 131 139 149 157 163 173 181 322 521 402 701 1001 1013 1207 1031 1339 1449 1527 1613 1743 1841 3222 5421)
elif [ "$PRIME" == "4" ]; then
  primes=(42 71 101 113 )
elif [ "$PRIME" == "5" ]; then
  primes=(127 131 139)
elif [ "$PRIME" == "6" ]; then
  primes=(149 157 163 173)
elif [ "$PRIME" == "7" ]; then
  primes=(181 322 521)
fi


echo "${primes[@]}"
network="$NET"
aug="$AUG"
saved_path_events="$SAVED_PATH_EVENTS"
saved_path_mi="$SAVED_PATH_MI"
saved_path_final="$SAVED_PATH_FINAL"
normalization="$NORM"
bandpass="$BANDPASS"
paradigm="$PARADIGM"

# /home/inbit/Scrivania/Datasets/2B/
# /mnt/datasets/eeg/Dataset_BCI_2b/Signals_BCI_2classes/

for seed in "${primes[@]}"; do
  # echo "Train Event seed: $seed"
  # python -u train_events.py --seed "$seed" --name_model "$network" --saved_path "$saved_path_events" --lr 0.001 \
  #         --augmentation "$aug" --num_workers 10 --normalization "$normalization" --paradigm "$paradigm" \
  #         --train_set "/mnt/beegfs/sbove/2B/3_classes/train_2b_$bandpass.npz" \
  #         --patience 150 --batch_size 72
#   echo "Train seed: $seed"
#   python -u train_motor_imagery.py --seed "$seed" --name_model "$network" --saved_path "$saved_path_mi" --lr 0.001 \
#           --augmentation "$aug" --num_workers 10 --normalization "$normalization" --paradigm "$paradigm" \
#           --train_set "/mnt/beegfs/sbove/2B/3_classes/train_2b_$bandpass.npz" \
#           --patience 150 --batch_size 72
  echo "Test seed: $seed"
  python -u test_motor_imagery_double_stage.py --name_model "$network" --saved_path_mi "$saved_path_mi" \
          --saved_path_events "$saved_path_events" --saved_path_final "$saved_path_final" --paradigm "$paradigm" \
          --test_set "/mnt/beegfs/sbove/2B/3_classes/test_2b_$bandpass.npz" --seed "$seed"
done

python create_excel_motor_imagery.py --network "$network" --path "$saved_path_final"
echo 'ok'