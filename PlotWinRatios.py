import matplotlib.pyplot as plt
import pickle
import numpy as np
from numpy.ma.extras import average



def entropy(p):
    return sum(-p*np.log2(p))


#Read the pickle file containing games
filename = '.pkl'
with open(filename, 'rb') as f:
    all_games = pickle.load(f)

chameleon_types = [matchup[0] for matchup in all_games.keys()]
chameleon_types = sorted(list(set(chameleon_types)))
truthful_types = [matchup[1] for matchup in all_games.keys()]
truthful_types = sorted(list(set(truthful_types)))

valid_table = np.zeros((len(chameleon_types), len(truthful_types)))
win_table = np.zeros((len(chameleon_types), len(truthful_types)))
identification_table = np.zeros((len(chameleon_types), len(truthful_types)))
second_stage_win_table = np.zeros((len(chameleon_types), len(truthful_types)))
for i in range(len(chameleon_types)):
    for j in range(len(truthful_types)):
        games = all_games[(chameleon_types[i], truthful_types[j])]
        valid_table[len(chameleon_types) - 1 - i][j] = games['num_of_valid_trials']/games['num_of_trials']
        win_table[len(chameleon_types) - 1 - i][j] = games['num_of_chameleon_loses']/games['num_of_valid_trials']
        identification_table[len(chameleon_types) - 1 - i][j] = games['num_of_chameleon_identified']/games['num_of_valid_trials']
        second_stage_win_table[len(chameleon_types) - 1 - i][j] = 1- games['num_of_chameleon_loses']/games['num_of_chameleon_identified']
#round to 2 decimal places
valid_table = np.round(valid_table, 2)
win_table = np.round(win_table, 2)
identification_table = np.round(identification_table, 2)
second_stage_win_table = np.round(second_stage_win_table, 2)

#create heatmap using seaborn
import seaborn as sns
sns.set_theme()
#Adjust the labels if not using these types
chameleon_types_short = ['Claude 3.5', 'Gemini 1.5', 'GPT 3.5', 'GPT 4', 'GPT 4o']
chameleon_types_short = chameleon_types_short[::-1]
truthful_types_short = ['Claude 3.5', 'Gemini 1.5', 'GPT 3.5', 'GPT 4', 'GPT 4o']


plt.figure(figsize=(4, 4))
ax = sns.heatmap(valid_table, annot=True, yticklabels=chameleon_types_short, xticklabels=truthful_types_short,vmin=0, vmax=1, square=True)
ax.set_title("Valid Games Ratio")
ax.set(ylabel="Chameleon type", xlabel="Non-chameleon type")
plt.subplots_adjust(left=-0, right=1, top=0.9, bottom=0.4)
plt.show()

plt.figure(figsize=(4, 4))
ax = sns.heatmap(win_table, annot=True, yticklabels=chameleon_types_short, xticklabels=truthful_types_short,vmin=0, vmax=1, square=True)
ax.set_title("Non-Chameleon Win Ratio")
ax.set(ylabel="Chameleon type", xlabel="Non-chameleon type")
plt.subplots_adjust(left=-0, right=1, top=0.9, bottom=0.4)
plt.show()

plt.figure(figsize=(4, 4))
ax = sns.heatmap(identification_table, annot=True, yticklabels=chameleon_types_short, xticklabels=truthful_types_short,vmin=0, vmax=1, square=True)
ax.set_title("Identification Ratio")
ax.set(ylabel="Chameleon type", xlabel="Non-chameleon type")
plt.subplots_adjust(left=-0, right=1, top=0.9, bottom=0.4)
plt.show()


plt.figure(figsize=(4, 4))

ax = sns.heatmap(second_stage_win_table, annot=True, yticklabels=chameleon_types_short, xticklabels=truthful_types_short,vmin=0, vmax=1, square=True)
ax.set_title("Second Chance Win Ratio")
ax.set(ylabel="Chameleon type", xlabel="Non-chameleon type")
plt.subplots_adjust(left=-0, right=1, top=0.9, bottom=0.4)
plt.show()
