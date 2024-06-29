#!/bin/bash

# Esegui lo script Python per creare un'istanza su Vast.ai o utilizzare un'istanza esistente
INSTANCE_ID=$(python src/vastai.py)

if [ -z "$INSTANCE_ID" ]; then
  echo "Failed to start or find an instance."
  exit 1
fi

vastai start instance --api-key "$(cat ~/.vast_api_key)" $INSTANCE_ID

while true; do
    STATE=$(vastai show instances --api-key "$(cat ~/.vast_api_key)" $INSTANCE_ID --raw | jq -r '.[0].state')
    if [ "$STATE" == "running" ]; then
        break
    fi
    echo "Waiting for instance to be in running state..."
    sleep 10
done

vastai copy $INSTANCE_ID train.sh :/workspace/train.sh

vastai exec $INSTANCE_ID -- bash /workspace/train.sh
