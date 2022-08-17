#!/bin/bash
# This script downloads the OnlyPlanes 1.0 dataset. 

#sudo apt-get install unzip

SYNTH_DATASET_URL="https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/OnlyPlanes_dataset_08122022.zip"

mkdir -p datasets
cd datasets

echo "=================================================="
echo "Downloading Microsoft Synthetic OnlyPlanes Dataset"

wget -c -O onlyplanes_dataset.zip "$SYNTH_DATASET_URL"
echo " ... unpacking OnlyPlanes synthetic data (it can take up to a few minutes)"
unzip -q onlyplanes_dataset.zip
rm onlyplanes_dataset.zip

echo "=================================================="
echo "Downloading complete"

