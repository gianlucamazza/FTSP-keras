#!/bin/bash

text="Verify the full code and help me to understand better the error.\n"



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
