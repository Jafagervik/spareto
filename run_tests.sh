#!/usr/bin/bash

# Run every helper file to check if the tests run

python ./helpers/utils.py
python ./helpers/datasetup.py
python ./helpers/metrics.py

echo "All tests ran!"
