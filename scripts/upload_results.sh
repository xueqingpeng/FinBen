#!/bin/bash

source .env
echo "HF_TOKEN: $HF_TOKEN"

python ./upload_results.py
