#!/bin/bash

# Delete all the screens
i=0
for filename in ./scripts/*.sh; do
    screen_name="scr${i}"
    i=$((i+1))
    screen -S "$screen_name" -p 0 -X break
    screen -S "$screen_name" -p 0 -X quit
done
