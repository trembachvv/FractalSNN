find . -type f -exec sh -c 'echo -e "\n\n--- FILE: {} ---"; cat "{}"' \; > all.txt
