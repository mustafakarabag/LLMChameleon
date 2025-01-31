import matplotlib.pyplot as plt
import pickle
import numpy as np
from numpy.ma.extras import average
from AbstractTalker.SCoRe.SCoRe import SCoRe
import traceback
from tqdm import tqdm

def entropy(p):
    return sum(-p*np.log2(p))


#Read the pickle file containing games
file_name = '.pkl'
with open(file_name , 'rb') as f:
    all_games = pickle.load(f)

compute_posterior_probabilities = True
for matchup in all_games.keys():
    #Choose a matchup to compute the info gains
    if matchup == ('GptTalker gpt-4o-2024-08-06', 'GptTalker gpt-4o-2024-08-06'):
        games = all_games[matchup]
        average_ent = []
        std_ent = []
        average_prob = []
        std_prob = []
        num_of_players = len(games['game_logs'][0]['word_responses'].keys())
        try:
            for trial in tqdm(range(games['num_of_trials']), desc='Posterior probabilities computed'):

                game_dict = games['game_logs'][trial]
                if game_dict['game_result'] != 'Fail':
                    secret_word_index = game_dict['possible_words'].index(game_dict['secret_word'])

                    # Create auxilary SCoRe player to compute posterior probabilities
                    aux_player = SCoRe()
                    aux_player.get_category(game_dict['category'], game_dict['possible_words'])

                    if compute_posterior_probabilities:


                        # Compute the posterior probabilities for each response
                        initial_prior_probabilities = np.ones(len(game_dict['possible_words'])) / len(game_dict['possible_words'])
                        probability_list = []
                        probability_list.append(initial_prior_probabilities)
                        responses_without_chameleon = [game_dict['word_responses'][j] for j in range(1, num_of_players + 1) if
                                                       j != game_dict['chameleon_index'] + 1]
                        for i in range(len(game_dict['word_responses']) - 1):

                            posterior_probabilities = aux_player.compute_posterior_probabilities(initial_prior_probabilities,
                                                                                                 responses_without_chameleon[
                                                                                                 :i + 1])
                            probability_list.append(posterior_probabilities)
                        all_games[matchup]['game_logs'][trial]['posterior_probabilities'] = probability_list
        except Exception as e:
            #Common exceptions include the LLMs generating outputs not allowed in the game rules
            print(e)
            print(traceback.format_exc())
            print("An error occurred for this gameplay. Ignoring the gameplay and continuing.")
            pass

import datetime
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
with open(f'{file_name} with posteriors.pkl', 'wb') as f:
    pickle.dump(all_games, f)
