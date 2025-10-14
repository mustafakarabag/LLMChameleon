import os
import re
import pickle
import random
from functools import partial
import numpy as np
import dill
import torch

from llmg.utils.llm import get_llm_response_and_hidden_states
from llmg.chameleon.constants import MSG_COLLECTION_POINTS, RESPOND_PROMPT
from llmg.chameleon.NaturalLanguageTalker.naturallanguagetalker import NaturalLanguageTalker


def construct_result_dict(game_logs):
    # Counts for the game result stats
    num_of_valid_trials = 0
    num_of_chameleon_identified = 0
    num_of_chameleon_loses = 0
    num_of_fails = 0

    # Result counter from the truthful players' pov
    for game_log in game_logs:
        if game_log['game_result'] == 'IdentifiedWin':
            num_of_chameleon_identified += 1
            num_of_chameleon_loses += 1
            num_of_valid_trials += 1
        elif game_log['game_result'] == 'IdentifiedLoss':
            num_of_chameleon_identified += 1
            num_of_valid_trials += 1
        elif game_log['game_result'] == 'MisidentifiedLoss':
            num_of_valid_trials += 1
        elif game_log['game_result'] == 'Fail':
            num_of_fails += 1
        else:
            raise ValueError(f"Unexpected game result: {game_log['game_result']}")

    all_results, results = {}, {}
    results['game_logs'] = game_logs
    results['num_of_players'] = len(game_logs[0]["player_types"])
    results['num_of_trials'] = len(game_logs)
    results['num_of_valid_trials'] = num_of_valid_trials
    results['num_of_chameleon_identified'] = num_of_chameleon_identified
    results['num_of_chameleon_loses'] = num_of_chameleon_loses
    results['num_of_fails'] = num_of_fails
    chameleon_idx = game_logs[0]["chameleon_index"]
    chameleon_type = game_logs[0]["player_types"][chameleon_idx]
    truthful_type = game_logs[0]["player_types"][(chameleon_idx - 1) % len(game_logs[0]["player_types"])]
    all_results[(chameleon_type, truthful_type)] = results

    return all_results


def load_game_logs(
    data_path,
    layer_to_probe=None,
    token_to_probe=None,
    max_games=None,
    verbose=2,
):
    """Load game data from a file or directory."""
    if verbose > 0:
        print(
            f"Loading data from {data_path}\n"
            f"  Probing layer: {layer_to_probe}\n"
            f"  Token position: {token_to_probe}"
        )

    if isinstance(data_path, list) or isinstance(data_path, tuple):
        all_game_logs = []
        for path in data_path:
            if verbose > 0:
                print(f"Loading data from {path}")
            all_game_logs.extend(load_game_logs(
                path,
                layer_to_probe=layer_to_probe,
                token_to_probe=token_to_probe,
                max_games=max_games,
                verbose=verbose,
            ))
            if max_games is not None and len(all_game_logs) >= max_games:
                break
    else:
        if os.path.isfile(data_path): # all games in a single file
            try:
                with open(data_path, "rb") as f:
                    all_game_logs = pickle.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load game logs from {data_path}: {e}")
                all_game_logs = []
            if max_games is not None:
                all_game_logs = all_game_logs[:max_games]
        elif os.path.isdir(data_path): # load individual game files
            all_game_logs = []
            file_list = os.listdir(data_path)
            file_list = [f for f in file_list if f not in ("partial_results.pkl", "final_results.pkl")]
            file_list.sort(key=lambda x: int(re.search(r'\d+', x).group()))  # Sort by numeric part in filename
            for f_i, file_name in enumerate(file_list):
                if verbose > 0 and (f_i + 1) % 100 == 0:
                    print(f"Loading file {f_i + 1}/{len(file_list)}: {file_name}")

                # Only process .pkl files
                if file_name.endswith(".pkl"):
                    # Load the game data
                    try:
                        with open(os.path.join(data_path, file_name), "rb") as f:
                            game_data = pickle.load(f)
                    except Exception as e:
                        print(f"[ERROR] Failed to load game data from {file_name}: {e}")
                        continue

                    # Select data to store
                    for player_idx in range(len(game_data["messages"])):
                        for msg_idx in range(len(game_data["messages"][player_idx])):
                            if "hidden_states" not in game_data["messages"][player_idx][msg_idx]:
                                if game_data["messages"][player_idx][msg_idx]["role"] == "assistant" \
                                    and "huggingface" in game_data["player_types"][player_idx].lower():
                                    print(f"[WARNING] No hidden states found for player {player_idx} message {msg_idx} in game {file_name}. Skipping.")
                                continue

                            # Hidden states shape: (num_tokens, num_layers, hidden_size)
                            token_idxs = np.arange(len(game_data["messages"][player_idx][msg_idx]["hidden_states"])) if token_to_probe is None else [token_to_probe]
                            layer_idxs = np.arange(len(game_data["messages"][player_idx][msg_idx]["hidden_states"][0])) if layer_to_probe is None else [layer_to_probe]
                            game_data["messages"][player_idx][msg_idx]["hidden_states"] = game_data["messages"][player_idx][msg_idx]["hidden_states"][token_idxs][:, layer_idxs, :]

                    all_game_logs.append(game_data)

                else:
                    print(f"[WARNING] Skipping non-pickle file: {file_name}")

                if max_games is not None and f_i > max_games:
                    break
        else:
            raise ValueError(f"Data path {data_path} is neither a file nor a directory.")

    if verbose > 0:
        print(f"Loaded {len(all_game_logs)} games from {data_path}")

    return all_game_logs


def load_game_states(data_path, max_game_states=None, verbose=1):
    all_game_states = []
    file_list = [f for f in os.listdir(data_path) if f not in ("config.pkl",)]
    file_list.sort(key=lambda x: int(re.search(r'\d+', x).group()))  # Sort by numeric part in filename
    for f_i, file_name in enumerate(file_list):
        if verbose > 0 and (f_i + 1) % 100 == 0:
            print(f"Loading file {f_i + 1}/{len(file_list)}: {file_name}")

        # Only process .pkl files
        if file_name.endswith(".pkl"):
            # Load the game data
            try:
                with open(os.path.join(data_path, file_name), "rb") as f:
                    game_state = pickle.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load game data from {file_name}: {e}")
                continue

            all_game_states.append(game_state)
        else:
            print(f"[WARNING] Skipping non-pickle file: {file_name}")

        if max_game_states is not None and f_i > max_game_states:
            break

    return all_game_states


# -------------- PROBE UTILS --------------


def prompt_llm_at_breakpoint(
    game_logs,
    model,
    tokenizer,
    breakpoint_fn,
    create_message_fn,
    create_additional_dict_to_save_fn=None,
    create_responses_for_logprobs_collection_fn=None,
    generation_kwargs=None,
    token_idx=None,
    layer_idx=None,
    get_all_logprobs=False,
    verbose=1,
):
    """
    Prompts an external LLM at a specific point in pre-collected game logs.

    This utility reconstructs the conversation history up to a specified breakpoint
    for each game, adds a custom prompt, and queries an LLM to get its
    response and hidden states.
    """
    all_results = []
    if verbose > 0:
        print(f"Starting LLM intervention on {len(game_logs)} games...")

    for game_i, game in enumerate(game_logs):
        if verbose > 0 and (game_i > 0 and (game_i + 1) % 10 == 0):
            print(f"Processing game {game_i + 1}/{len(game_logs)}...")

        # 1. Determine the breakpoint for this game using the provided function
        breakpt = breakpoint_fn(game)
        if breakpt is None:
            if verbose > 1:
                print(f"  Skipping game {game_i} as no breakpoint was returned.")
            continue
        player_idx, msg_idx = breakpt # player idx is 0-based indexed

        # 2. Validate breakpoint and reconstruct the conversation history
        try:
            # The history is taken from the perspective of the player at the breakpoint
            conversation_history = game["messages"][player_idx][:msg_idx + 1]
        except IndexError:
             if verbose > 0:
                print(f"[WARNING] Invalid breakpoint {breakpt} for game {game_i}. Player or message index out of range. Skipping.")
             continue

        # 3. Generate the custom prompt
        custom_msg = create_message_fn(locals())

        # 4. Add the custom prompt to the conversation history to form the final input
        messages_for_llm = conversation_history + custom_msg

        # 5. Prepare response for which to collect log probabilities
        get_logprobs_of_responses = create_responses_for_logprobs_collection_fn(locals()) \
            if create_responses_for_logprobs_collection_fn is not None else None

        # 6. Construct the result entry
        result_entry = {
            "game_index": game_i,
            # "category": game["category"],
            # "possible_words": game["possible_words"],
            # "secret_word": game["secret_word"],
            # "votes": game["votes"],
            # "word_responses": game["word_responses"],
            "breakpoint": breakpt,
            "messages": messages_for_llm,
        }

        # 7. Query the LLM
        try:
            if isinstance(model, NaturalLanguageTalker):
                model.clear_messages()  # Clear previous messages in the talker
                model.messages = messages_for_llm[:-1]  # Set the current messages
                assistant_reply, updated_messages = model.send_message_get_response(
                    utterance=messages_for_llm[-1]["content"],
                )
                result_entry.update({
                    "response": assistant_reply,
                    **updated_messages[-1],  # Last message is the assistant's response
                })
            else:
                chat_template_kwargs = {
                    "return_tensors": "pt",
                    "add_generation_prompt": True,
                    "return_dict": True,
                    **({"enable_thinking": False} if "qwen" in model.config._name_or_path.lower() else {})
                }
                full_gen_kwargs = {
                    "pad_token_id": tokenizer.eos_token_id,
                    "output_hidden_states": True,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
                if generation_kwargs:
                    full_gen_kwargs.update(generation_kwargs)
                response_all = get_llm_response_and_hidden_states(
                    messages=messages_for_llm,
                    model=model,
                    tokenizer=tokenizer,
                    generation_kwargs=full_gen_kwargs,
                    chat_template_kwargs=chat_template_kwargs,
                    hidden_state_token_idxs=token_idx,
                    hidden_state_layer_idxs=layer_idx,
                    get_logprobs_of_responses=get_logprobs_of_responses,
                    get_all_logprobs=get_all_logprobs,
                )
                if get_logprobs_of_responses is None or len(get_logprobs_of_responses) == 0:
                    # response, hidden_states, logprobs, logprobs_argmax = response_all
                    result_entry.update({
                        "response": response_all["content"],
                        "hidden_states": response_all["hidden_states"],  # Shape: (num_tokens, num_layers, hidden_size)
                        "logprobs": response_all["logprobs"],  # Shape: (num_tokens, vocab_size) or (num_tokens,)
                        "logprobs_argmax": response_all["logprobs_argmax"],  # Shape: (num_tokens,)
                    })
                else:
                    # Collecting logprobs for specific responses
                    result_entry["responses"] = response_all["logprobs"]
        except Exception as e:
            if verbose > 0:
                print(f"[ERROR] Failed to get LLM response for game {game_i}: {e}")
            continue

        # 7. Store the results for this game
        result_entry.update(
            **((create_additional_dict_to_save_fn(locals()) or {}) if create_additional_dict_to_save_fn is not None else {}),
        )
        all_results.append(result_entry)

    if verbose > 0:
        print(f"Finished processing. Collected results for {len(all_results)} games.")

    return all_results


def collect_llm_preds_at_breakpoints(
    game_logs,
    model,
    tokenizer,
    token_idx,  # Index of the token to extract hidden states for
    layer_idx,  # Index of the layer to extract hidden states for
    all_other_player_idxs, # # Indices for selecting response words of players (0-based indexing)
    remap_player_ids_to_be_increasing=True, # When all_other_player_idxs contain decreasing sequences, remap the player IDs in the prompt to be increasing 
    generation_kwargs=None,
    include_chameleon_response=False,  # Whether to include the chameleon's response in the prompt
    free_generation=False,  # Whether to let the LLM generate freely
    collect_logprobs_of_all_secret_words=False,  # Whether to collect logprobs of all possible secret words or just the correct one
    collect_logprobs_of_capitalized_secret_words=True,  # Whether to collect logprobs of capitalized secret words
    verbose=True,
):
    assert not (free_generation and collect_logprobs_of_all_secret_words), "Incompatible settings"
    all_results = []

    def breakpoint_fn(game_log):
        if game_log.get("word_responses", None) is None:
            if verbose:
                print(f"[WARNING] Game has no word responses. Skipping.")
            return None
        
        # Get the message index and player index for the breakpoint when to collect LLM predictions
        msg_idx = MSG_COLLECTION_POINTS["last-game-setup-confirmation"]["msg_idx"][0]
        player_idx = game_log["chameleon_index"]
        return player_idx, msg_idx

    for other_player_idxs in all_other_player_idxs:
        
        def create_message_fn(locs):
            messages_to_add = []

            # Add chameleon's response if requested (user-assistant message pair)
            cham_index = locs["game"]["chameleon_index"] # 0-indexed
            if include_chameleon_response and cham_index in other_player_idxs:
                all_chameleon_messages = locs["game"]["messages"][cham_index]
                cham_response_prompt = RESPOND_PROMPT.format(previous_words="", instruction="").strip()
                messages_to_add.extend([
                    {"role": "user", "content": cham_response_prompt},
                    all_chameleon_messages[MSG_COLLECTION_POINTS["my-word"]["msg_idx"][0]],
                ])

            # Collect response words from selected players
            word_responses = locs["game"]["word_responses"]
            chameleon_index = locs["game"]["chameleon_index"]
            others_said, player_ids_to_assign, other_idx = [None] * len(other_player_idxs), [None] * len(other_player_idxs), 0
            for player_id, word in word_responses.items():
                # player_id uses 1-based indexing
                if player_id == (chameleon_index + 1) and not include_chameleon_response:
                    continue
                if other_idx in other_player_idxs:
                    # Place in the correct order or what was said, specified by the order in other_player_idxs
                    player_ids_to_assign[other_player_idxs.index(other_idx)] = player_id
                    others_said[other_player_idxs.index(other_idx)] = word
                other_idx += 1

            # Construct player prompts
            if remap_player_ids_to_be_increasing:
                # Sort only player_ids_to_assign (change what player ID said what)
                player_ids_to_assign = sorted(player_ids_to_assign, key=lambda x: int(x))
            for i, (player_id, word) in enumerate(zip(player_ids_to_assign, others_said)):
                player_said_prompt = f"Player {player_id}{' (you)' if player_id == (chameleon_index + 1) else ''} said '{word}'"
                others_said[i] = player_said_prompt

            # Combine the responses into a prompt
            if len(others_said) == 0:
                others_said = "Nothing has been said yet"
            else:
                others_said = ". ".join(others_said)
            prompt = (
                f"Here are what the players said: {others_said}.\n"
                f"Based on what the other players have said so far and based on the possible secret words, what do you think the secret word is? Remember, all players except you know the secret word. Say the exact secret word that you believe they are concealing. Do not say anything else."
            )
            messages_to_add.append({"role": "user", "content": prompt})

            return messages_to_add

        def create_additional_dict_to_save_fn(locs):
            # Extract the hidden states from the last non-chameleon's response
            others_idxs = [
                i for i in range(len(locs["game"]["messages"]))
                if i != locs["game"]["chameleon_index"]
            ]
            last_nonchameleon_idx = [i for _i, i in enumerate(others_idxs) if _i in other_player_idxs]
            if len(last_nonchameleon_idx) > 0:
                last_nonchameleon_idx = last_nonchameleon_idx[-1]
                last_nonchameleon_msgs = locs["game"]["messages"][last_nonchameleon_idx]
                last_nonchameleon_hs = last_nonchameleon_msgs[MSG_COLLECTION_POINTS["my-word"]["msg_idx"][0]].get("hidden_states", None)
            else:
                last_nonchameleon_hs = None  # Return empty tensor if no valid index
                last_nonchameleon_idx = None

            return {
                "last_nonchameleon_hs": last_nonchameleon_hs,  # Shape: (num_layers, hidden_size)
                "last_nonchameleon_idx": last_nonchameleon_idx,
                "other_player_idxs": other_player_idxs,
            }

        def create_responses_for_logprobs_collection_fn(locs):
            # If free generation is enabled, we do not collect logprobs for specific responses
            if free_generation:
                return None
            
            # Collect logprobs of pre-defined responses
            collect_logprobs_of = []
            if collect_logprobs_of_all_secret_words:
                possible_words = locs["game"]["possible_words"]
                collect_logprobs_of.extend(possible_words)
                if collect_logprobs_of_capitalized_secret_words:
                    for word in possible_words:
                        collect_logprobs_of.append(word.capitalize()) # only first word has first letter capitalized
                        collect_logprobs_of.append(" ".join([w.capitalize() for w in word.split(" ")])) # all words have first letter capitalized
            else:
                collect_logprobs_of.append(locs["game"]["secret_word"])
                if collect_logprobs_of_capitalized_secret_words:
                    collect_logprobs_of.append(locs["game"]["secret_word"].capitalize())
                    collect_logprobs_of.append(" ".join([w.capitalize() for w in locs["game"]["secret_word"].split(" ")]))
            collect_logprobs_of = list(set(collect_logprobs_of))  # Remove duplicates in case of single-word secret words

            return collect_logprobs_of

        all_results.append(prompt_llm_at_breakpoint(
            game_logs=game_logs,
            model=model,
            tokenizer=tokenizer,
            breakpoint_fn=breakpoint_fn,
            create_message_fn=create_message_fn,
            create_additional_dict_to_save_fn=create_additional_dict_to_save_fn,
            generation_kwargs=generation_kwargs,
            create_responses_for_logprobs_collection_fn=create_responses_for_logprobs_collection_fn,
            token_idx=token_idx,  # Index of the token to extract hidden states for
            layer_idx=layer_idx,  # Index of the layer to extract hidden states for
            verbose=verbose,
        ))

    return all_results
