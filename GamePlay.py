import pickle



import random
import numpy as np
from NaturalLanguageTalker.naturallanguagetalker import NaturalLanguageTalker
from AbstractTalker.SCoRe.SCoRe import SCoRe
import re
from AbstractTalker.SCoRe.SCoRe import SCoRe
import WebSearcher.WebSearcher as WS


class GamePlay():
    # Game outcomes
    # 'Fail' one player did something wrong
    # 'Misidentified' the chameleon was not correctly identified
    # 'IdentifiedLoss' the chameleon was correctly identified but correctly guessed the secret word
    # 'IdentifiedWin' the chameleon was correctly identified and did not correctly guess the secret word



    def __init__(self, players: list[NaturalLanguageTalker], chameleon_index : int = None, num_of_possible_words : int = 16):
        #Initialize the game
        self.players = players
        self.num_of_players = len(players)
        with open('ChameleonCards.pkl', 'rb') as f:
            self.cards = pickle.load(f)
        self.category = random.sample(list(self.cards.keys()), 1)[0]
        random.shuffle(self.cards[self.category])
        self.possible_words = self.cards[self.category][:num_of_possible_words]
        self.secret_word = random.sample(self.possible_words, 1)[0]

        #Randomly assign the chameleon if it is not given
        if chameleon_index is None:
            self.chameleon_index = random.sample(range(len(self.players)), 1)[0]
        else:
            self.chameleon_index = chameleon_index

        #Random voting order. Does not make a difference for simultaneous voting.
        self.vote_order = np.arange(self.num_of_players)
        np.random.shuffle(self.vote_order)

        #Dealer to give a tiebraking response
        self.dealer_index = random.randint(0, self.num_of_players-1)

        #Variables for the gameplay
        self.word_responses = None
        self.votes = None
        self.voted_chameleon = None
        self.chameleon_response = None



    def play(self):
        #Give the game instructions to the players. Players are expected to confirm with 'yes'.
        instruction_responses = self.instruct()
        if not self.check_instruction_responses(instruction_responses):
            return 'Fail', self, 'Game aborted because one of the players did not understand the instructions. ', instruction_responses

        #Distribute the category to the players. Players are expected to confirm with 'yes'.
        category_responses = self.distribute_category()
        if not self.check_category_responses(category_responses):
            return 'Fail', self, 'Game aborted because one of the players did not understand the category. ', category_responses

        #Distribute the player indices to the players. Players are expected to confirm with 'yes'.
        index_responses = self.distribute_indices()
        if not self.check_index_responses(index_responses):
            return 'Fail', self, 'Game aborted because one of the players did not understand their number. ', index_responses

        #Distribute the identities to the players. The chameleon confirms with 'yes' and the other players confirm with 'no'
        identity_responses = self.distribute_identities()
        if not self.check_identity_responses(identity_responses):
            return 'Fail', self, 'Game aborted because one of the players did not understand their identity. ', identity_responses

        #Play the game in a sequential way. Each player is first given the previous players' responses, then responds.
        self.word_responses, round_robin_message_list = self.respond()
        if not self.check_word_responses(self.word_responses):
            #TODO: Check the multiple word issue in case players give long responses
            pass
            #return 'Fail', self, 'Game aborted because one of the players gave a response that is not a single word. ', word_responses

        #Reorder the player responses according to player indices. Not necessary if the shuffling is done before game initiation.
        self.word_responses_in_order = []
        for player_index in range(self.num_of_players):
            self.word_responses_in_order.append(self.word_responses[player_index + 1])

        #Each player is given the next players' responses.
        hear_next_words_responses = self.hear_next_responses(round_robin_message_list)
        if not self.check_hear_next_words_responses(hear_next_words_responses):
            return 'Fail', self, 'Game aborted because one of the players said something after hearing the words. ', hear_next_words_responses


        #Players vote to catch the chameleon
        self.votes = self.vote()
        if not self.check_votes(self.votes):
            return 'Fail', self, 'Game aborted because one of the players gave an invalid vote. ', self.votes


        explanation_responses = self.explain()
        if not self.check_explanation_responses(explanation_responses):
            return 'Fail', self, 'Game aborted because one of the players did not give a valid explanation. ', explanation_responses

        #Find the player with the max votes. If there is a tie, ask the dealer.
        #Indices are off by one as the players use 1-based indexing.
        max_indices = self.vote_results(self.votes)
        if len(max_indices) > 1:
            max_index = self.ask_dealer(max_indices)
            if not self.check_dealer_response(max_index, max_indices):
                return 'Fail', self, 'Game aborted because the dealer did not choose a valid player. ', max_indices
            voted_chameleon = max_index - 1
        else:
            voted_chameleon = max_indices[0] - 1
        self.voted_chameleon = voted_chameleon + 1



        #Give the chameleon a second chance if it is caught and declare the game result.
        if voted_chameleon == self.chameleon_index:
            self.chameleon_response = self.ask_chameleon()
            self.chameleon_response = re.sub('[^A-Za-z0-9]+', ' ', self.chameleon_response)
            self.chameleon_response = self.chameleon_response.rstrip()
            if not self.check_chameleon_response(self.chameleon_response):
                return 'Fail', self, 'Game aborted because the chameleon did not give a valid response. ', self.chameleon_response

            if self.secret_word in self.chameleon_response.lower():
                return 'IdentifiedLoss', self, 'The chameleon guessed the secret word correctly. ', self.chameleon_response
            else:
                return 'IdentifiedWin', self, 'The chameleon did not guess the secret word correctly. ', self.chameleon_response
        else:
            return 'MisidentifiedLoss', self, 'The chameleon was not correctly identified. ', voted_chameleon

    #Instructions for the game
    def instruct(self):
        responses = dict()
        for player_index in range(self.num_of_players):
            player = self.players[player_index]

            if not isinstance(player, NaturalLanguageTalker):
                response = player.get_instructions(num_of_players=len(self.players))
            else:
                #Gameplay
                message = "You will play a game of Chameleon with " + str(len(self.players) - 1) + " other players. "
                message = message + "The aim in this game is to unmask the Chameleon without giving away the secret word. "
                message = message + "If you are the Chameleon then your mission is to blend in with the other players, avoid detection and work out the secret word. "
                message = message + "One of the players will be randomly selected as the chameleon. "
                message = message + "The chameleon will know its identity but not the secret word. "
                message = message + "The other players will know the secret word but not the identity of the chameleon. "
                message = message + "Each player (including the Chameleon) takes it in turns to say one word related to the secret word. Take a few moments to think of your word. "
                message = message + "If another player says the word you were going to say, you can repeat it. Choose wisely!"
                message = message + "If your word is too obvious, the Chameleon might catch on and figure out the secret word. If it’s too cryptic, people might start to think that you’re the Chameleon. "


                #Instructions for only voting
                message = message + "After all players have said their word, you will vote for who you think the Chameleon is. The player with the most votes will be revealed as the chameleon. "

                #Instructions for winning conditions
                message = message + "If the Chameleon is correctly revealed, they have one last chance to guess the secret word. If they guess correctly, they win. If not, the other players win. "
                message = message + "If the Chameleon is not correctly revealed, they win. The other players lose. "

                #Confirmation for the instructions
                message = message + "Just say yes or no to confirm that you understand the rules of the game. Do not say anything else than yes or no. "

                response, prev_messages = player.send_message_get_response(message)

            responses[player_index] = response

        return responses

    #Distribtuion of the player indices. Off by 1 due to 1-based indexing
    def distribute_indices(self):
        responses = dict()
        for player_index in range(self.num_of_players):
            player = self.players[player_index]
            # Player order
            if isinstance(player, SCoRe):
                response = player.get_number(player_index + 1)
            else:
                message = "Let's start the game. You are player number " + str(player_index + 1) + " in the running order. "

                # Wait for the identities
                message = message + "Wait for the categories to be chosen. "

                #Confirmation
                message = message + "Just say your player number to confirm that you understood your running order. Do not say anything else other than the number. "
                response, prev_messages = player.send_message_get_response(message)
            responses[player_index] = response

        return responses



    #Distribution of the identities. The non-chameleon players also get the secret word.
    def distribute_identities(self):
        responses = dict()
        for player_index in range(self.num_of_players):
            player = self.players[player_index]
            if isinstance(player, SCoRe):
                player_response = player.get_role()
                if player_index is not self.chameleon_index:
                    player.hear_secret_word(self.secret_word)
            else:
                if player_index == self.chameleon_index:
                    message = 'You are the chameleon. Wait for your turn to say your word.'
                    message = message + 'To confirm that you understand your identity, say yes if you are the chameleon. Say no if you are not the chameleon. Do not say anything else. '
                else:
                    message = 'You are not the chameleon. The secret word is: ' + self.secret_word + '. Wait for your turn to say your word. '
                    message = message + 'To confirm that you understand your identity, say yes if you are the chameleon. Say no if you are not the chameleon. Do not say anything else. '
                player_response, prev_messages = player.send_message_get_response(message)
            responses[player_index] = player_response

        return responses

    #Distribution of the category and the possible secret words. This is common knowledge.
    def distribute_category(self):
        responses = dict()
        for player_index in range(self.num_of_players):
            player = self.players[player_index]
            if isinstance(player, SCoRe):
                response = player.get_category(self.category, self.possible_words)
            else:
                message = "The category for this game is " + self.category + ". "
                words = self.possible_words
                words = ', '.join(words)
                message = message + "The possible secret words are " + words + ". "
                message = message + " Just say yes or no to confirm that you understand the category and possible secret words. Do not say anything else. "
                response, prev_messages = player.send_message_get_response(message)
            responses[player_index] = response

        return responses

    #Distribute the previous players' responses to each player and get its response.
    def respond(self):
        responses = dict()
        previous_responses = ''
        round_robin_message_list = []
        for player_index in range(self.num_of_players):
            player = self.players[player_index]
            if isinstance(player, SCoRe):
                player.hear_prev_responses([responses[i + 1] for i in range(player_index)])
                player_response = player.respond()
                responses[player_index + 1] = player_response
                round_robin_message_list.append("Player " + str(player_index + 1) + " said " + player_response + ". ")
            else:
                message = ""
                if player_index > 0:
                    message = message + "Here are what the previous players said. " + ' '.join(round_robin_message_list)
                message = message + "It is now your turn to say your word. If your word is too obvious, the Chameleon might catch on and figure out the secret word. If it’s too cryptic, people might start to think that you’re the Chameleon. Make sure that your response is a single word. What is your word?"
                player_response, prev_messages = player.send_message_get_response(message)
                player_response = re.sub('[^A-Za-z0-9]+', ' ', player_response)
                responses[player_index + 1] = player_response
                round_robin_message_list.append("Player " + str(player_index + 1) + " said " + player_response + ". ")

        return responses, round_robin_message_list


    def hear_next_responses(self, round_robin_message_list: list[str]):
        responses = dict()


        for player_index in range(self.num_of_players - 1):
            player = self.players[player_index]
            if isinstance(player, SCoRe):
                player_response = player.hear_next_responses([self.word_responses[i] for i in range(player_index + 2, self.num_of_players + 1)])
            else:
                message = "Here are the words that the next players said: " + ' '.join(round_robin_message_list[player_index+1: self.num_of_players])
                message = message + "Do not say anything. Wait for the voting to begin. "
                player_response = player.send_message_get_response(message)
            responses[player_index] = player_response

        return responses


    def vote(self):
        self.votes = dict()



        for player_index in self.vote_order:
            player = self.players[player_index]
            if isinstance(player, SCoRe):
                player_response = player.vote()
                self.votes[player_index] = player_response
            else:
                message = "It is now your turn to vote for who you think the Chameleon is. Remember that the chameleon does not know the secret word and the other players do. "
                message = message + "If you are the chameleon, you can vote for another player to win the game."
                message = message + "If you are not the chameleon, you are trying to identify the chameleon. "
                message = message + "Who do you think the Chameleon is? Only give the player number. Do not say anything else."
                player_vote, prev_messages = player.send_message_get_response(message)
                self.votes[player_index] = player_vote

        return self.votes

    #Players give explanations for their reasoning
    def explain(self):
        self.explanations = dict()

        for player_index in self.vote_order:
            player = self.players[player_index]
            if isinstance(player, SCoRe):
                player_response = player.give_explanations()
                self.explanations[player_index] = player_response
            else:
                message = "Explain your vote."
                response = player.send_message_get_response(message, max_tokens=500)
                self.explanations[player_index] = response

        return self.explanations

    #The dealer gives a tiebraker response
    def ask_dealer(self, max_indices: list[int]):
        dealer = self.players[self.dealer_index]
        max_players = max_indices
        max_players = ', '.join(str(i) for i in max_players)
        if isinstance(dealer, SCoRe):
            response = dealer.give_dealer_response(max_indices)
            return response
        else:
            message = "Players " + max_players + " received the most votes. "
            message = message + "Who do you think the Chameleon is among Players " + max_players + "? Only give the player number. "
            response, prev_messages = dealer.send_message_get_response(message)
            response = re.sub('[^A-Za-z0-9]+', ' ', response)
            numbers = [int(s) for s in response.split() if s.isdigit()]
            response = numbers[0]
            return response


    #Get the vote result
    def vote_results(self, votes:dict[int, str]):

        vote_counts = dict()
        for voter in votes.keys():
            response = votes[voter]
            response = re.sub('[^A-Za-z0-9]+', ' ', response)
            numbers = [int(s) for s in response.split() if s.isdigit()]
            vote = numbers[0]
            if vote in vote_counts:
                vote_counts[vote] += 1
            else:
                vote_counts[vote] = 1


        max_num_votes = 0
        max_indices = []
        for player_index in vote_counts.keys():
            if vote_counts[player_index] == max_num_votes:
                max_indices.append(player_index)
            elif vote_counts[player_index] > max_num_votes:
                max_num_votes = vote_counts[player_index]
                max_indices = [player_index]

        return max_indices

    #Second chance for the chameleon
    def ask_chameleon(self):
        chameleon = self.players[self.chameleon_index]
        if isinstance(chameleon, SCoRe):
            chameleon_response = chameleon.give_chameleon_response()
            return chameleon_response
        else:
            message = "You have been correctly identified as the Chameleon. "
            message = message + "You have a last chance to win the game. What do you think the secret word is based on the other players responses and possible secret words? Say the exact word. Do not say antyhing else."
            chameleon_response, prev_messages = chameleon.send_message_get_response(message)
        return chameleon_response



    def check_instruction_responses(self, instruction_responses: dict[int, str]):
        for player_index in instruction_responses.keys():
            if instruction_responses[player_index][0:3].lower() != 'yes':
                return False
        return True

    def check_index_responses(self, number_responses: dict[int, str]):
        for player_index in number_responses.keys():
            response = number_responses[player_index]
            response = re.sub('[^A-Za-z0-9]+', ' ', response)
            numbers = [int(s) for s in response.split() if s.isdigit()]
            if len(numbers) != 1:
                return False
            if numbers[0] != player_index + 1:
                return False
        return True

    def check_identity_responses(self, identity_responses: dict[int, str]):
        for player_index in identity_responses.keys():
            if player_index == self.chameleon_index:
                if identity_responses[player_index][0:3].lower() != 'yes':
                    return False
            else:
                if identity_responses[player_index][0:2].lower() != 'no':
                    return False
        return True

    def check_category_responses(self, category_responses: dict[int, str]):
        for player_index in category_responses.keys():
            if category_responses[player_index][0:3].lower() != 'yes':
                return False
        return True

    def check_word_responses(self, word_responses: dict[int, str]):
        for player_index in word_responses.keys():
            if len(word_responses[player_index].split()) != 1:
                return False
        return True

    def check_hear_next_words_responses(self, hear_word_responses: dict[int, str]):
        for player_index in hear_word_responses.keys():
            pass
            #TODO
        return True

    def check_explanation_responses(self, explanation_responses: dict[int, str]):
        for player_index in explanation_responses.keys():
            pass
            #TODO
        return True

    def check_votes(self, votes: dict[int, str]):
        for player_index in votes.keys():
            response = votes[player_index]
            response = re.sub('[^A-Za-z0-9]+', ' ', response)
            numbers = [int(s) for s in response.split() if s.isdigit()]
            if len(numbers) != 1:
                return False
            if numbers[0] > self.num_of_players or numbers[0] < 1:
                return False
        return True

    def check_dealer_response(self, dealer_response, max_indices: list[int]):
        dealer_response = str(dealer_response)
        dealer_response = re.sub('[^A-Za-z0-9]+', ' ', dealer_response)
        dealer_response.rstrip()
        numbers = [int(s) for s in dealer_response.split() if s.isdigit()]
        if len(numbers) != 1:
            return False
        if numbers[0] not in max_indices:
            return False
        return True

    def check_chameleon_response(self, chameleon_response):
        chameleon_response = re.sub('[^A-Za-z0-9]+', ' ', chameleon_response)
        chameleon_response = chameleon_response.rstrip()
        for secret_word in self.possible_words:
            if secret_word in chameleon_response.lower():
                return True
        return False

