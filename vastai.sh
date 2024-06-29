#!/bin/bash

pip install vastai

INSTANCE=11240109

# Connect
ssh "$(vastai ssh-url $INSTANCE)"

