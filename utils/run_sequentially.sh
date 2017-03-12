#!/bin/bash

# Make sure everything in scripts is executable
chmod +x ./scripts/*.sh

# Run files one by one
for filename in ./scripts/*.sh; do
    $filename
done
