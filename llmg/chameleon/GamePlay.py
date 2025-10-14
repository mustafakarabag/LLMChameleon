import pickle
import random
import numpy as np
import re

from llmg.chameleon.NaturalLanguageTalker.naturallanguagetalker import NaturalLanguageTalker
from llmg.chameleon.constants import (
    GAME_START_PROMPT,
    DISTRIBUTE_INDICES_PROMPT,
    DISTRIBUTE_CHAMELEON_IDENTITY_PROMPT,
    DISTRIBUTE_NON_CHAMELEON_IDENTITY_PROMPT,
    DISTRIBUTE_CATEGORY_PROMPT,
    RESPOND_PROMPT,
    HEAR_NEXT_RESPONSES_PROMPT,
    VOTE_PROMPT,
    EXPLAIN_VOTE_PROMPT,
    ASK_DEALER_PROMPT,
    CHAMELEON_SECOND_CHANCE_PROMPT
)


class GamePlay():
    # Game outcomes
    # 'Fail' one player did something wrong
    # 'Misidentified' the chameleon was not correctly identified
    # 'IdentifiedLoss' the chameleon was correctly identified but correctly guessed the secret word
    # 'IdentifiedWin' the chameleon was correctly identified and did not correctly guess the secret word

    def __init__(
        self,
        players,
        chameleon_cards_path,
        chameleon_index=None,
        num_of_possible_words=16,
        init_from_ckpt=None,
        nonchameleons_response_word_instruction=None, # additional instructions for non-chameleons to give a response word
    ):
        # Initialize the game
        self.players = players
        self.num_of_players = len(players)
        with open(chameleon_cards_path, 'rb') as f:
            self.cards = pickle.load(f)
        self.nonchameleons_response_word_instruction = nonchameleons_response_word_instruction
        params_loaded_from_ckpt = []

        # Category
        if init_from_ckpt is not None and "category" in init_from_ckpt:
            self.category = init_from_ckpt['category']
            params_loaded_from_ckpt.append('category')
        else:
            # Initialize the game state randomly
            self.category = random.sample(list(self.cards.keys()), 1)[0]
        
        # Possible words
        if init_from_ckpt is not None and "possible_words" in init_from_ckpt:
            self.possible_words = init_from_ckpt['possible_words']
            params_loaded_from_ckpt.append('possible_words')
        else:
            random.shuffle(self.cards[self.category])
            self.possible_words = self.cards[self.category][:num_of_possible_words]

        # Secret word
        if init_from_ckpt is not None and "secret_word" in init_from_ckpt:
            self.secret_word = init_from_ckpt['secret_word']
            params_loaded_from_ckpt.append('secret_word')
        else:
            self.secret_word = random.sample(self.possible_words, 1)[0]

        # Chameleon index
        if chameleon_index is None:
            if init_from_ckpt is not None and "chameleon_index" in init_from_ckpt:
                self.chameleon_index = init_from_ckpt['chameleon_index']
                params_loaded_from_ckpt.append('chameleon_index')
            else:
                # Randomly assign the chameleon if it is not given
                self.chameleon_index = random.sample(range(len(self.players)), 1)[0]
        else:
            self.chameleon_index = chameleon_index

        # Voting order
        self.vote_order = None
        if init_from_ckpt is not None and ("vote_order" in init_from_ckpt or "votes" in init_from_ckpt):
            if init_from_ckpt.get("vote_order") is None:
                if init_from_ckpt.get("votes") is not None:
                    self.vote_order = list(init_from_ckpt["votes"].keys())
            else:
                self.vote_order = init_from_ckpt["vote_order"]
        if self.vote_order is None:
            # Random voting order. Does not make a difference for simultaneous voting.
            self.vote_order = np.arange(self.num_of_players)
            np.random.shuffle(self.vote_order)
        else:
            params_loaded_from_ckpt.append('vote_order')

        # Dealer index
        if init_from_ckpt is not None and "dealer_index" in init_from_ckpt:
            self.dealer_index = init_from_ckpt['dealer_index']
            params_loaded_from_ckpt.append('dealer_index')
        else:
            # Dealer to give a tiebraking response
            self.dealer_index = random.randint(0, self.num_of_players - 1)
        
        print(f"Game initialized with parameters from checkpoint: {params_loaded_from_ckpt}")

        # Variables for the gameplay
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
            self.chameleon_response = re.sub('[^A-Za-z0-9-.]+', ' ', self.chameleon_response)
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
                message = GAME_START_PROMPT.format(num_players=len(self.players) - 1)
                response, prev_messages = player.send_message_get_response(message)

            responses[player_index] = response

        return responses

    #Distribtuion of the player indices. Off by 1 due to 1-based indexing
    def distribute_indices(self):
        responses = dict()
        for player_index in range(self.num_of_players):
            player = self.players[player_index]
            message = DISTRIBUTE_INDICES_PROMPT.format(player_number=str(player_index + 1))
            response, prev_messages = player.send_message_get_response(message)
            responses[player_index] = response

        return responses

    #Distribution of the identities. The non-chameleon players also get the secret word.
    def distribute_identities(self):
        responses = dict()
        for player_index in range(self.num_of_players):
            player = self.players[player_index]
            if player_index == self.chameleon_index:
                message = DISTRIBUTE_CHAMELEON_IDENTITY_PROMPT
            else:
                message = DISTRIBUTE_NON_CHAMELEON_IDENTITY_PROMPT.format(secret_word=self.secret_word)
            player_response, prev_messages = player.send_message_get_response(message)
            responses[player_index] = player_response

        return responses

    #Distribution of the category and the possible secret words. This is common knowledge.
    def distribute_category(self):
        responses = dict()
        for player_index in range(self.num_of_players):
            player = self.players[player_index]
            words = self.possible_words
            words = ', '.join(words)
            message = DISTRIBUTE_CATEGORY_PROMPT.format(category=self.category, possible_words=words)
            response, prev_messages = player.send_message_get_response(message)
            responses[player_index] = response

        return responses

    #Distribute the previous players' responses to each player and get its response.
    def respond(self):
        responses = dict()
        round_robin_message_list = []
        for player_index in range(self.num_of_players):
            player = self.players[player_index]
            previous_responses = ''
            if player_index > 0:
                previous_responses = "Here are what the previous players said. " + ' '.join(round_robin_message_list)
            instruction = ""
            if player_index != self.chameleon_index and self.nonchameleons_response_word_instruction is not None:
                instruction = self.nonchameleons_response_word_instruction
            message = RESPOND_PROMPT.format(previous_words=previous_responses, instruction=instruction).strip()
            player_response, prev_messages = player.send_message_get_response(message)
            player_response = re.sub('[^A-Za-z0-9-.]+', ' ', player_response)
            responses[player_index + 1] = player_response
            round_robin_message_list.append("Player " + str(player_index + 1) + " said " + player_response + ".")

        return responses, round_robin_message_list

    def hear_next_responses(self, round_robin_message_list: list[str]):
        responses = dict()
        for player_index in range(self.num_of_players - 1):
            player = self.players[player_index]
            message = HEAR_NEXT_RESPONSES_PROMPT.format(next_words=' '.join(round_robin_message_list[player_index + 1: self.num_of_players]))
            player.add_message(message, send_with_next_message=True)
            responses[player_index] = ""
        return responses

    def vote(self):
        self.votes = dict()
        for player_index in self.vote_order:
            player = self.players[player_index]
            player_vote, prev_messages = player.send_message_get_response(VOTE_PROMPT)
            self.votes[player_index] = player_vote
        return self.votes

    #Players give explanations for their reasoning
    def explain(self):
        self.explanations = dict()
        for player_index in self.vote_order:
            player = self.players[player_index]
            response, _ = player.send_message_get_response(EXPLAIN_VOTE_PROMPT, max_tokens=3000)
            self.explanations[player_index] = response
        return self.explanations

    #The dealer gives a tiebraker response
    def ask_dealer(self, max_indices: list[int]):
        dealer = self.players[self.dealer_index]
        max_players = max_indices
        max_players = ', '.join(str(i) for i in max_players)
        message = ASK_DEALER_PROMPT.format(max_players=max_players)
        response, prev_messages = dealer.send_message_get_response(message)
        response = re.sub('[^A-Za-z0-9]+', ' ', response)
        numbers = [int(s) for s in response.split() if s.isdigit()]
        response = numbers[0] if len(numbers) > 0 else response
        return response

    #Get the vote result
    def vote_results(self, votes:dict[int, str]):
        # Count the votes
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

        # Find the player with the most votes
        # If there is a tie, return all players with the most votes
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
        chameleon_response, prev_messages = chameleon.send_message_get_response(CHAMELEON_SECOND_CHANCE_PROMPT)
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
        chameleon_response = re.sub('[^A-Za-z0-9-.]+', ' ', chameleon_response)
        chameleon_response = chameleon_response.rstrip()
        for secret_word in self.possible_words:
            if secret_word in chameleon_response.lower():
                return True
        return False
