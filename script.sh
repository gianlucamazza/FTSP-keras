#!/bin/bash

text="Verify and optimize the following project. Show me only the most important mistakes or optimizations of the code.\n"



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
