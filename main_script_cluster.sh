#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --ntasks=1
#SBATCH --account=lm_foggia
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=07:00:00
#SBATCH --nodelist=gnode07
#SBATCH --job-name=MSVT_SE_Net_1
#SBATCH --output=MSVT_SE_Net_1.log
#SBATCH --dependency=113895

export TORCH_DEVICE=cuda
export PYTHON=/home/bvosmn000/.conda/envs/ICareMeEnv/bin/python

if [ -z "$NET" ] || [ -z "$PRIME" ] || [ -z "$AUG" ] || [ -z "$SAVED_PATH" ] || [ -z "$NORM" ] || [ -z "$BANDPASS" ] || [ -z "$PARADIGM" ] || [ -z "$CLASSES" ]; then
    echo "Errore: Devi specificare NET, PRIME, AUG, SAVED_PATH, NORM, BANDPASS, PARADIGM, CLASSES!"
    echo "Utilizzo: NET=<valore> PRIME=<valore> AUG=<valore> ./script.sh"
    exit 1
fi

echo "$NET"
echo "$PRIME"
echo "$AUG"
echo "$SAVED_PATH"
echo "$NORM"
echo "$BANDPASS"
echo "$PARADIGM"
echo "$TORCH_DEVICE"
echo "$CLASSES"

if [ "$PRIME" == "1" ]; then
  primes=(42 71 101 113 127 131 139 149 157 163 173 181 322 521)
elif [ "$PRIME" == "2" ]; then
  primes=(402 701 1001 1013 1207 1031 1339 1449 1527 1613 1743 1841 3222 5421)
elif [ "$PRIME" == "3" ]; then
  primes=(42 71 101 113 127 131 139 149 157 163 173 181 322 521 402 701 1001 1013 1207 1031 1339 1449 1527 1613 1743 1841 3222 5421)
elif [ "$PRIME" == "4" ]; then
  primes=(42 71 101 113)
elif [ "$PRIME" == "5" ]; then
  primes=(127 131 139)
elif [ "$PRIME" == "6" ]; then
  primes=(149 157 163 173)
elif [ "$PRIME" == "7" ]; then
  primes=(521)
fi


echo "${primes[@]}"
network="$NET"
aug="$AUG"
saved_path="$SAVED_PATH"
normalization="$NORM"
bandpass="$BANDPASS"
paradigm="$PARADIGM"
classes="$CLASSES"

for seed in "${primes[@]}"; do
  echo "Train seed: $seed"
  $PYTHON -u train_motor_imagery.py --seed "$seed" --name_model "$network" --saved_path "$saved_path" --lr 0.001 \
          --augmentation "$aug" --num_workers 32 --normalization "$normalization" --paradigm "$paradigm" \
          --train_set "/mnt/beegfs/sbove/2B/${classes}_classes/train_2b_$bandpass.npz" --device "$TORCH_DEVICE"\
          --patience 100 --batch_size 72
  $PYTHON -u test_motor_imagery.py --name_model "$network" --saved_path "$saved_path" --paradigm "$paradigm" \
          --test_set "/mnt/beegfs/sbove/2B/${classes}_classes/test_2b_$bandpass.npz" --device "$TORCH_DEVICE"\
          --seed "$seed"
done

$PYTHON create_excel_motor_imagery.py --network "$network" --path "$saved_path"
echo 'ok'
