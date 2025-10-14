# Prompts and constants for the Chameleon game
GAME_START_PROMPT = """
You will play a game of Chameleon with {num_players} other players. The aim in this game is to unmask the Chameleon without giving away the secret word. If you are the Chameleon, your mission is to blend in with the other players, avoid detection, and work out the secret word. One of the players will be randomly selected as the Chameleon. The Chameleon will know its identity but not the secret word. The other players will know the secret word but will not know who the Chameleon is. Each player (including the Chameleon) takes it in turns to say one word related to the secret word. Take a few moments to think of your word. If another player says the word you were going to say, you can repeat it. Choose wisely! If your word is too obvious, the Chameleon might catch on and figure out the secret word. If it's too cryptic, people might start to think that you're the Chameleon. After all players have said their word, you will vote for who you think the Chameleon is. The player with the most votes will be revealed as the Chameleon. If the Chameleon is correctly revealed, the Chameleon has one last chance to guess the secret word. If the Chameleon guesses correctly, the Chameleon wins. If not, the other players win. If the Chameleon is not correctly revealed, the Chameleon wins. The other players lose. Just say yes or no to confirm that you understand the rules of the game. Do not say anything else than yes or no.
""".strip()

DISTRIBUTE_INDICES_PROMPT = """
Let's start the game. You are player number {player_number} in the running order. Wait for the identities to be chosen. Just say your player number to confirm that you understood your running order. Do not say anything else other than the number.
""".strip()

DISTRIBUTE_CHAMELEON_IDENTITY_PROMPT = """
You are the Chameleon. Wait for your turn to say your word. To confirm that you understand your identity, say yes if you are the Chameleon. Say no if you are not the Chameleon. Do not say anything else.
""".strip()

DISTRIBUTE_NON_CHAMELEON_IDENTITY_PROMPT = """
You are not the Chameleon. The secret word is: {secret_word}. Wait for your turn to say your word. To confirm that you understand your identity, say yes if you are the Chameleon. Say no if you are not the Chameleon. Do not say anything else.
""".strip()

DISTRIBUTE_CATEGORY_PROMPT = """
The category for this game is {category}. The possible secret words are {possible_words}. Just say yes or no to confirm that you understand the category and possible secret words. Do not say anything else.
""".strip()

RESPOND_PROMPT = """
{previous_words} It is now your turn to say your word. If your word is too obvious, the Chameleon might catch on and figure out the secret word. If it's too cryptic, people might start to think that you're the Chameleon. {instruction}Make sure that your response is a single word. What is your word?
""".strip()

HEAR_NEXT_RESPONSES_PROMPT = """
Here are the words that the next players said: {next_words}
""".strip()

VOTE_PROMPT = """
It is now your turn to vote for who you think the Chameleon is. Remember that the Chameleon does not know the secret word and the other players do. If you are the Chameleon, you can vote for another player to win the game. If you are not the Chameleon, you are trying to identify the Chameleon. Who do you think the Chameleon is? Only give the player number. Do not say anything else.
""".strip()

EXPLAIN_VOTE_PROMPT = """
Explain your vote.
""".strip()

ASK_DEALER_PROMPT = """
Players {max_players} received the most votes. Who do you think the Chameleon is among Players {max_players}? Only give the player number.
""".strip()

CHAMELEON_SECOND_CHANCE_PROMPT = """
You have been correctly identified as the Chameleon. You have a last chance to win the game. What do you think the secret word is based on the other players' responses and possible secret words? Say the exact word. Do not say anything else.
""".strip()

MSG_COLLECTION_POINTS = { # 0-indexed
    "last-game-setup-confirmation": {
        "msg_idx": (7,),
    },
    "my-word": {
        "msg_idx": (9,),
        "last_msg_contains": "what is your word?",
        "select_fn": lambda locs: locs["player_idx"] != locs["game"]["chameleon_index"]
    },
    "last-word": {
        "msg_idx": (9,),
        "last_msg_contains": "what is your word?",
        "select_fn": lambda locs: locs["player_idx"] != locs["game"]["chameleon_index"] \
                                    and locs["player_idx"] == locs["num_players"] - 1
    },
    "voting": {
        "msg_idx": (11, 13),
        "last_msg_contains": "who do you think the chameleon is?",
        "select_fn": lambda locs: locs["player_idx"] != locs["game"]["chameleon_index"]
    },
    "voting-with-chameleon": {
        "msg_idx": (11, 13),
        "last_msg_contains": "who do you think the chameleon is?",
    },
}
