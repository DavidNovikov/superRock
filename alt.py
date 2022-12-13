import os

rockTypes = ['g', 'm', 'sa', 'sl', 'ag']
colors = ['white', 'grey', 'black', 'yellow', 'red', 'pink', 'green', 'blue']

for rock in rockTypes:
    for color in colors:
        with open(f'txtFiles/{rock}_{color}.txt', 'w') as f:
            f.write('')
            f.close()