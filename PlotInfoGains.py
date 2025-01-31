
import matplotlib.pyplot as plt
import pickle
import numpy as np
from numpy.ma.extras import average
from AbstractTalker.SCoRe.SCoRe import SCoRe
import traceback
from tqdm import tqdm



def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def entropy(p):
    return sum(-p*np.log2(p))


#Read the pickle file containing games
file_name = '.pkl'
with open(file_name, 'rb') as f:
    all_games = pickle.load(f)

average_ent = []
std_ent = []
average_prob = []
std_prob = []
avg_is_max = []
avg_is_max_2 = []
avg_is_max_3 = []
avg_is_max_4 = []

for matchup in all_games.keys():
    #Choose a matchup to plot
    if matchup == ('GptTalker gpt-4o-2024-08-06', 'GptTalker gpt-4o-2024-08-06'):
        games = all_games[matchup]
        for index_wo_chameleon in range(games['num_of_players']):
            ent_list = []
            prob_list = []
            is_max_list = []
            is_max_2_list = []
            is_max_3_list = []
            is_max_4_list = []
            for trial in range(games['num_of_trials']):
                game_dict = games['game_logs'][trial]
                if game_dict['game_result'] != 'Fail':
                    secret_word_index = game_dict['possible_words'].index(game_dict['secret_word'])
                    prob = game_dict['posterior_probabilities'][index_wo_chameleon]
                    is_max = np.argmax(prob) == secret_word_index
                    is_max_2 = secret_word_index in np.argsort(prob)[-2:]
                    is_max_3 = secret_word_index in np.argsort(prob)[-3:]
                    is_max_4 = secret_word_index in np.argsort(prob)[-4:]
                    is_max_list.append(is_max)
                    is_max_2_list.append(is_max_2)
                    is_max_3_list.append(is_max_3)
                    is_max_4_list.append(is_max_4)
                    ent_list.append(entropy(prob))
                    prob_of_secret_word = game_dict['posterior_probabilities'][index_wo_chameleon][secret_word_index]
                    prob_list.append(prob_of_secret_word)

            average_ent.append(np.mean(ent_list))
            std_ent.append(np.std(ent_list))
            average_prob.append(np.mean(prob_list))
            std_prob.append(np.std(prob_list))
            avg_is_max.append(np.mean(is_max_list))
            avg_is_max_2.append(np.mean(is_max_2_list))
            avg_is_max_3.append(np.mean(is_max_3_list))
            avg_is_max_4.append(np.mean(is_max_4_list))

        # Plotting the average entropy and std deviation entropy
        fig, ax = plt.subplots()
        ax.errorbar(range(1, games['num_of_players'] + 1), average_ent, yerr=std_ent, fmt='o')
        ax.set_xlabel('Response index')
        ax.set_xticks(range(1,5))
        ax.set_xticklabels(['Prior', '1', '2', '3'])
        ax.set_ylabel('Average entropy (bits)')
        ax.set_yticks(np.arange(0, 4.5, 1))
        plt.subplots_adjust(left=0.3, right=0.7, top=0.7, bottom=0.3)

        plt.show()



        plt.show()
        fig, ax = plt.subplots()

        ax.plot(range(1, games['num_of_players'] + 1), avg_is_max, 'o')
        #plot the average probability of the secret word being one of the two most probable
        ax.plot(range(1, games['num_of_players'] + 1), avg_is_max_2, 'o')
        #plot the average probability of the secret word being one of the three most probable
        ax.plot(range(1, games['num_of_players'] + 1), avg_is_max_3, 'o')
        #plot the average probability of the secret word being one of the four most probable
        ax.plot(range(1, games['num_of_players'] + 1), avg_is_max_4, 'o')

        #Add text k=1 next to the last data point of the first plot
        ax.text(4.15, avg_is_max[-1], r'$k=1$', verticalalignment='center')
        #Add text k=2 next to the last data point of the second plot
        ax.text(4.15, avg_is_max_2[-1], r'$k=2$', verticalalignment='center')
        #Add text k=3 next to the last data point of22222 the third plot
        ax.text(4.15, avg_is_max_3[-1], r'$k=3$', verticalalignment='center')
        #Add text k=4 next to the last data point of the fourth plot
        ax.text(4.15, avg_is_max_4[-1], r'$k=4$', verticalalignment='center')



        ax.set_xlabel('Response index')
        plt.xlim([0.8, 5])
        ax.set_xticks(range(1,5))
        ax.set_xticklabels(['Prior', '1', '2', '3'])
        plt.ylim([0, 0.81])
        ax.set_yticks(np.arange(0, 0.81, 0.2))
        ax.set_ylabel(r'Probability')
        plt.subplots_adjust(left=0.3, right=0.7, top=0.7, bottom=0.3)

        plt.show()