#!/usr/bin/bash

# Install path to where images are stored

pip3 install kaggle --upgrade

# Or alternatively, load through different means

# Download train and test, move to folder and move
unzip data.zip
mv data/* data/imgs
rm data.zip
