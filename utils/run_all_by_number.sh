#!/bin/bash

getScreenName() {
    echo ${1##*/}
}

# Make sure everything in scripts is executable
chmod +x ./scripts/*.sh

# Create screens
i=0
for filename in ./scripts/*.sh; do
    screen_name="scr${i}"
    i=$((i+1))
    (sleep 1; screen -d;) &
    screen -S "$screen_name"
done

# Run something in each screen
i=0
for filename in ./scripts/*.sh; do
    COMMAND="$filename"
    screen_name="scr${i}"
    i=$((i+1))
    screen -S "$screen_name" -X stuff "$COMMAND"$'\n'
done
