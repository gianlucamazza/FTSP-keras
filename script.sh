#!/bin/bash

text="Check the whole project and help me to create a perfect train.py.\n"
text+="- the dataset should be split into windows of 60 days for training and 30 days for validation, overlapping by 30 days.\n"
text+="show me the full code.\n"


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
) | xclip -selection clipboard
