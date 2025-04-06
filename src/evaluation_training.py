'''
    THIS META-AGENT TRAINING-EVALUATION IS IMPLEMENTED FOR THE LUNAR LANDER SETUP
    During training, Weights & Biases was kept active for live monitoring,
    and at the end the output.log file was saved to support own training evaluation.
'''

import re
import numpy as np

# path to the training output file from wandb
file_path = "../policies/meta_choice_02.log"

# read all lines
with open(file_path, "r") as file:
    lines = file.readlines()

# output file for evaluation
output_file = "training_choice_02.log"

# lists to store temp and mean values
temp_values = []
mean_values = []

temp = 1

# search flag to distinguish between evaluation and training episodes
search = False

for line in lines:

    # match episode number
    episode_match = re.search(r"'episode_counter':\s*(\d+)", line)

    # check if the episode matches
    if episode_match and int(episode_match.group(1)) >= temp:

        # if exactly matched, +1 for next episode
        if int(episode_match.group(1)) == temp:
            temp += 1
        # due to step-based training, as explained in my thesis, there are jumps in episode numbers,
        # which result in skipped episode numbers.
        # using the '> temp' condition I capture these episode numbers
        elif int(episode_match.group(1)) > temp:
            temp += 2

        search = True

    # match meta reward value
    value_match = re.search(r"'meta_reward': array\(\[(-?[\d\.]+)\]", line)

    # save in the list if match
    if value_match and search:

        search = False
        if value_match:
            value = float(value_match.group(1))
            print(value)
            temp_values.append(value)

            # calculate mean-reward every 10 episodes
            if len(temp_values) == 10:
                mean_values.append(np.mean(temp_values))
                temp_values = []

# print and save
print(mean_values)
with open(output_file, "w") as outfile:
    for mean_values in mean_values:
        print(np.around(mean_values.item(), 2), file=outfile)