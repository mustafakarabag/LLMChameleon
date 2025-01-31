import random
import traceback
import pickle
from NaturalLanguageTalker.GptTalker import GptTalker
from NaturalLanguageTalker.ClaudeTalker import ClaudeTalker
from NaturalLanguageTalker.GeminiTalker import GeminiTalker
from AbstractTalker.SCoRe.SCoRe import SCoRe
from GamePlay import GamePlay
from tqdm import tqdm
import numpy as np


# Game outcomes
# 'Fail' one player did something wrong
# 'Misidentified' the chameleon was not correctly identified
# 'IdentifiedLoss' the chameleon was correctly identified but correctly guessed the secret word
# 'IdentifiedWin' the chameleon was correctly identified and did not correctly guess the secret word

num_of_trials = 100
num_of_players = 4

#Add any new player types here. Include the model_id if applicable, if not it should be None.
chameleons = [[GptTalker, "gpt-4o-2024-08-06"], [GptTalker, "gpt-4-0613"],  [GptTalker, "gpt-3.5-turbo-0125"], [ClaudeTalker, "claude-3-5-sonnet-20241022"], [GeminiTalker, "gemini-1.5-pro"], [ChameleonSCoRe, None]]
truthfuls = [[GptTalker, "gpt-4o-2024-08-06"], [GptTalker, "gpt-4-0613"],  [GptTalker, "gpt-3.5-turbo-0125"], [ClaudeTalker, "claude-3-5-sonnet-20241022"], [GeminiTalker, "gemini-1.5-pro"], [TruthfulSCoRe, None]]

#Choose the indices of the models to be tested
chameleon_type_index_list = [0,1,2,3,4]
truthful_type_index_list = [0,1,2,3,4]
tested_chameleon_types = [chameleons[index] for index in chameleon_type_index_list]
tested_truthful_types = [truthfuls[index] for index in truthful_type_index_list]

#Dictionary to store the results. Each key corresponds to a chameleon-truthful player type combination
all_results = dict()

#Set to True to the responses for each game
print_individual_games = True

#Set True to compute the posterior probabilities for each response
compute_posterior_probabilities = False

# Set the chameleon truthful player combination
for chameleon_type in tested_chameleon_types:
    for truthful_type in tested_truthful_types:
        #Counts for the game result stats
        num_of_valid_trials = 0
        num_of_chameleon_identified = 0
        num_of_chameleon_loses = 0



        #Dictionary to store the results of each combination
        results = dict()
        #List to keep each game log
        game_logs = []

        print(f'Chameleon type: {chameleon_type[0].__name__} {chameleon_type[1]}, Truthful type: {truthful_type[0].__name__} {truthful_type[1]}')
        for run in tqdm(range(num_of_trials), desc='Games played'):
            #Exception block to exclude invalid games
            try:
                #Determine the chameleon index uniformly randomly
                chameleon_index = random.randint(0, num_of_players-1)

                # Initiate players based on identities
                players = []
                for i in range(num_of_players):
                    if i == chameleon_index:
                        if chameleon_type[1] is not None:
                            players.append(chameleon_type[0](model_id=chameleon_type[1]))
                        else:
                            players.append(chameleon_type[0]())
                    else:
                        if truthful_type[1] is not None:
                            players.append(truthful_type[0](model_id=truthful_type[1]))
                        else:
                            players.append(truthful_type[0]())

                #Initiate conversation
                for player in players:
                    player.start_conversation()

                #Create the game. The category and the secret word are chosen
                game = GamePlay(players, chameleon_index=chameleon_index, num_of_possible_words=16)





                #Game result
                game_result, game, explanation, last_responses = game.play()

                if compute_posterior_probabilities:
                    #Create auxilary SCoRe player to compute posterior probabilities
                    aux_player = SCoRe()
                    aux_player.get_category(game.category, game.possible_words)
                    aux_player.responses = [game.word_responses[i] for i in range(1, game.num_of_players+1)]

                    #Compute the posterior probabilities for each response
                    initial_prior_probabilities = np.ones(len(game.possible_words)) / len(game.possible_words)
                    probability_list = []
                    probability_list.append(initial_prior_probabilities)
                    for i in range(len(game.word_responses) - 1):
                        responses_without_chameleon = [game.word_responses[j] for j in range(1, game.num_of_players+1) if j != game.chameleon_index+1]
                        posterior_probabilities = aux_player.compute_posterior_probabilities(initial_prior_probabilities, responses_without_chameleon[:i+1])
                        probability_list.append(posterior_probabilities)

                    print(probability_list)




                #Result counter from the truthful players' pov
                if game_result == 'IdentifiedWin':
                    num_of_chameleon_identified += 1
                    num_of_chameleon_loses += 1
                    num_of_valid_trials += 1
                elif game_result == 'IdentifiedLoss':
                    num_of_chameleon_identified += 1
                    num_of_valid_trials += 1
                elif game_result == 'MisidentifiedLoss':
                    num_of_valid_trials += 1

                if print_individual_games:
                    print("=================================================================================")
                    print('Game result:' +game_result)
                    print('Explanation: ' + explanation)
                    print('Category: ' + game.category)
                    print('Possible words: ' + str(game.possible_words))
                    print('Secret word: ' + game.secret_word)
                    print('Chameleon index: ' +str(game.chameleon_index+1))
                    print('Player types: ' + str([player.__class__.__name__ + ' ' + player.model_id for player in game.players]))
                    print('Game word responses' + str(game.word_responses))
                    print('Votes' + str(game.votes))
                    print('Voted chameleon: ' + str(game.voted_chameleon))
                    print('Chameleon response: ' + str(game.chameleon_response))
                    print('\n')
                    print(f'Game played {run+1} times.')
                    print(f'Valid games: {num_of_valid_trials} out of {run+1}')
                    print(f'Number of times the chameleon was identified: {num_of_chameleon_identified}')
                    print(f'Number of times the chameleon loses: {num_of_chameleon_loses}')
                    print('\n')
                    for player in game.players:
                        print(player.messages)
                    print("=================================================================================")

                game_dict = {'game_result': game_result, 'explanation': explanation, 'category': game.category, 'possible_words': game.possible_words, 'secret_word': game.secret_word, 'chameleon_index': game.chameleon_index, 'word_responses': game.word_responses, 'votes': game.votes, 'voted_chameleon': game.voted_chameleon, 'chameleon_response': game.chameleon_response, 'messages': [player.messages for player in game.players]}
                if compute_posterior_probabilities:
                    game_dict['posterior_probabilities'] = probability_list
                

                game_dict['player_types'] = [player.__class__.__name__ + ' ' +  player.model_id for player in game.players]
                game_logs.append(game_dict)


            except Exception as e:
                #Common exceptions include the LLMs generating outputs not allowed in the game rules
                print(e)
                print(traceback.format_exc())
                print("An error occurred for this gameplay. Ignoring the gameplay and continuing.")
                pass

        #Print results for the chameleon-truthful player combination
        print(f'Chameleon type: {chameleon_type[0].__name__} {chameleon_type[1]}, Truthful type: {truthful_type[0].__name__} {truthful_type[1]}')
        print(f'Number of games: {num_of_trials}')
        print(f'Valid games: {num_of_valid_trials}')
        print(f'Number of times the chameleon was identified: {num_of_chameleon_identified}')
        print(f'Number of times the chameleon loses: {num_of_chameleon_loses}')
        if num_of_valid_trials > 0:
            print(f'Non-chameleon win ratio: {num_of_chameleon_loses/num_of_valid_trials}')
            print(f'Identification ratio {num_of_chameleon_identified/num_of_valid_trials}')
        if num_of_chameleon_identified > 0:
            print(f'Second round win ratio {num_of_chameleon_loses/num_of_chameleon_identified}')
        print('\n')

        #Save the results in the dictionary
        results['game_logs'] = game_logs
        results['num_of_players'] = num_of_players
        results['num_of_trials'] = num_of_trials
        results['num_of_valid_trials'] = num_of_valid_trials
        results['num_of_chameleon_identified'] = num_of_chameleon_identified
        results['num_of_chameleon_loses'] = num_of_chameleon_loses
        all_results[(chameleon_type[0].__name__ + ' ' +  str(chameleon_type[1]), truthful_type[0].__name__ + ' ' + str(truthful_type[1]) )] = results


        #Save the game logs in a pickle file with timestamp
        import datetime
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        with open(f'Partial Results_{timestamp}.pkl', 'wb') as f:
            pickle.dump(all_results, f)

#Save the game logs in a pickle file with timestamp

now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
with open(f'Final Results_{timestamp}.pkl', 'wb') as f:
    pickle.dump(all_results, f)

