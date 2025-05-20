#!/bin/bash

if [ ! -d "data" ]; then
    gdown 1LDXeed4Ymke8FhB8inK9l5Y1ZcHC3uLN -O data.zip
    unzip data.zip
fi