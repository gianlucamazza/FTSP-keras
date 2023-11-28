#!/bin/bash

text="You are an expert in machine learning.\n"
text+="Analyze carefully the following tensorflow project for predicting the bitcoin price in the future and provide all code fixes and improvements using a snippet. Comment only with code.\n"

separator="\n\`\`\`\n"

(
    echo -e "$text"
    echo -e "data_preparation.py:"
    echo -e "$separator"
    cat data_preparation.py
    echo -e "$separator"
    echo -e "feature_engineering.py:"
    echo -e "$separator"
    cat feature_engineering.py
    echo -e "$separator"
    echo -e "model.py:"
    echo -e "$separator"
    cat model.py
    echo -e "$separator"
    echo -e "train.py:"
    echo -e "$separator"
    cat train.py
    echo -e "$separator"
    echo -e "predict.py:"
    echo -e "$separator"
    cat predict.py
    echo -e "$separator"
) | xclip -selection clipboard
