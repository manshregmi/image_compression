#!/bin/bash
# retry_download.sh

FILE_ID="17THA1IiPStSO6jG4h5clwkw0ySzgLZI"
OUTPUT_FILE="checkpoints/TCM_MSE_lambda_0.0067.pth.tar"
RETRY_DELAY=1800 # 30 minutes in seconds

while true; do
    echo "Attempting to download $OUTPUT_FILE..."
    if gdown "$FILE_ID" -O "$OUTPUT_FILE"; then
        echo "Download successful!"
        break
    else
        echo "Download failed. Retrying in $((RETRY_DELAY / 60)) minutes..."
        sleep $RETRY_DELAY
    fi
done
