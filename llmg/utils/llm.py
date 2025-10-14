import os
import torch
import numpy as np
from copy import deepcopy


def select_hidden_states(hidden_states, token_idxs=None, layer_idxs=None):
    """
    Selects specific token and layer indices from the hidden states tensor.

    Args:
        hidden_states: A tensor of shape (num_tokens, num_layers, hidden_size).
        token_idxs: A list of token indices to select. If None, selects all tokens.
        layer_idxs: A list of layer indices to select. If None, selects all layers.

    Returns:
        A tensor containing the selected hidden states.
    """
    if token_idxs is not None and layer_idxs is not None:
        _layer_idxs = [layer_idxs] if isinstance(layer_idxs, int) else layer_idxs
        _token_idxs = [token_idxs] if isinstance(token_idxs, int) else token_idxs
        return hidden_states[_token_idxs][:, _layer_idxs]
    elif token_idxs is not None:
        _token_idxs = [token_idxs] if isinstance(token_idxs, int) else token_idxs
        return hidden_states[_token_idxs]
    elif layer_idxs is not None:
        _layer_idxs = [layer_idxs] if isinstance(layer_idxs, int) else layer_idxs
        return hidden_states[:, _layer_idxs]
    else:
        return hidden_states


def get_llm_response_and_hidden_states(
    model,
    tokenizer,
    messages,
    chat_template_kwargs=None,
    generation_kwargs=None,
    additional_generation_kwargs_hook_fns=None,
    max_new_tokens=None,
    return_logprobs=False,
    get_logprobs_of_responses=None,
    get_all_logprobs=False,
    hidden_state_token_idxs=None,
    hidden_state_layer_idxs=None,
):
    if get_all_logprobs:
        return_logprobs = True

    dict_to_return = dict()
    _chat_template_kwargs = deepcopy(chat_template_kwargs or {})
    _chat_template_kwargs["return_dict"] = True
    _chat_template_kwargs["return_tensors"] = "pt"
    inputs = tokenizer.apply_chat_template(messages, **_chat_template_kwargs).to(model.device)
    inputs_len = inputs.input_ids.shape[1]
    
    # Run the model
    if get_logprobs_of_responses is None or len(get_logprobs_of_responses) == 0:
        # Generate response
        with torch.no_grad():
            # Set generation kwargs
            gen_kwargs = deepcopy(generation_kwargs) if generation_kwargs else dict()
            if max_new_tokens is not None:
                gen_kwargs["max_new_tokens"] = max_new_tokens
            if additional_generation_kwargs_hook_fns is not None:
                for hook_fn in additional_generation_kwargs_hook_fns:
                    gen_kwargs.update(hook_fn(locals()))
            gen_kwargs["output_hidden_states"] = True
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

            # Run the model
            outputs = model.generate(**inputs, **gen_kwargs)

        # Collect hidden states
        hs_list = []
        for step_hs in outputs.hidden_states:
            hs_list.append(torch.stack(step_hs, dim=1)[:, :, -1, :])
        hidden_states = torch.stack(hs_list, dim=1).detach().cpu()[0] # (num_tokens, num_layers, hidden_size)

        # Extract the specific token and layer hidden state
        hidden_states = select_hidden_states(hidden_states=hidden_states, token_idxs=hidden_state_token_idxs, layer_idxs=hidden_state_layer_idxs)
        dict_to_return["hidden_states"] = hidden_states  # Shape: (num_tokens, num_layers, hidden_size)

        # Decode response
        response_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
        dict_to_return["content"] = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        # Prepare logprobs if requested
        if return_logprobs:
            logprobs = torch.log_softmax(torch.stack(outputs.scores).squeeze(1).detach(), dim=-1)  # Shape: (num_tokens, vocab_size)
            logprobs_argmax = logprobs.argmax(dim=-1)  # Shape: (num_tokens,)
            if not get_all_logprobs:  # Collect logprobs only for the response tokens
                logprobs = logprobs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)  # Shape: (num_tokens,)
            dict_to_return["logprobs"] = logprobs.cpu()
            dict_to_return["logprobs_argmax"] = logprobs_argmax.cpu()

    # Allow just collecting logprobs on the provided response (no generation)
    elif get_logprobs_of_responses is not None and len(get_logprobs_of_responses) > 0:
        dict_to_return["logprobs"] = dict()

        # Pre-compute KV cache for the prompt
        with torch.no_grad():
            prompt_outputs = model(**inputs, use_cache=True, output_hidden_states=True, logits_to_keep=1)
            past_key_values = prompt_outputs.past_key_values
            prompt_hidden_states = torch.stack(prompt_outputs.hidden_states).squeeze(1).permute(1, 0, 2).cpu() # (num_tokens, num_layers, hidden_size)
            prompt_last_logits = prompt_outputs.logits[:, [-1], :]  # (batch_size, 1, vocab_size)

        for get_logprobs_of_response in get_logprobs_of_responses:
            # Tokenize only the response
            response_inputs = tokenizer(get_logprobs_of_response, return_tensors="pt", add_special_tokens=False).to(model.device)

            # Use KV cache for the forward pass on the response tokens
            with torch.no_grad():
                outputs = model(
                    input_ids=response_inputs.input_ids,
                    attention_mask=torch.cat([inputs.attention_mask, response_inputs.attention_mask], dim=1),
                    past_key_values=deepcopy(past_key_values),
                    output_hidden_states=True,
                    return_dict=True
                )

            # The logits for the response tokens are at the end of the sequence
            logprobs = torch.log_softmax(torch.cat((prompt_last_logits, outputs.logits[:, :-1]), dim=1), dim=-1)
            logprobs_argmax = logprobs.argmax(dim=-1).cpu()  # (batch_size, num_tokens)
            logprobs = logprobs.gather(dim=-1, index=response_inputs.input_ids.unsqueeze(-1))[0].squeeze(-1).cpu()  # (batch_size, num_tokens)
            response_text = tokenizer.decode(response_inputs.input_ids[0], skip_special_tokens=True).strip()

            # Concatenate hidden states from prompt and response
            response_hidden_states = torch.stack(outputs.hidden_states).squeeze(1).permute(1, 0, 2).cpu() # (num_tokens, num_layers, hidden_size)
            hidden_states = torch.cat((prompt_hidden_states[[-1]], response_hidden_states), dim=0)

            # Extract the specific token and layer hidden state
            hidden_states = select_hidden_states(hidden_states=hidden_states, token_idxs=hidden_state_token_idxs, layer_idxs=hidden_state_layer_idxs)

            # Store the results
            dict_to_return["logprobs"][get_logprobs_of_response] = {
                "response_text": response_text,
                "hidden_states": hidden_states,  # Shape: (num_tokens, num_layers, hidden_size)
                "logprobs": logprobs,  # Shape: (num_tokens,)
                "logprobs_argmax": logprobs_argmax,  # Shape: (num_tokens,)
            }
        
        # Select the response with the highest logprob as the main response ("content")
        if len(dict_to_return["logprobs"]) > 0:
            best_response = max(dict_to_return["logprobs"].items(), key=lambda x: x[1]["logprobs"].sum())
            dict_to_return["content"] = best_response[1]["response_text"]
        else:
            dict_to_return["content"] = None

    return dict_to_return
