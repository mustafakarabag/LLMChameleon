import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmg.chameleon.NaturalLanguageTalker.naturallanguagetalker import NaturalLanguageTalker
from llmg.utils.steering import SteeringHook
from llmg.utils.llm import get_llm_response_and_hidden_states


class HuggingfaceTalker(NaturalLanguageTalker):
    """Class to handle conversations for Huggingface players."""

    def __init__(
        self,
        model_id,
        model=None,
        tokenizer=None,
        additional_generation_kwargs=None,
        additional_generation_kwargs_hook_fns=None,
        hidden_states_layer_idx=None,
        hidden_states_token_idx=None,
        start_conversation=False,
        steering_kwargs=None,
    ):
        super().__init__(model_id=model_id, start_conversation=start_conversation)

        # Load model and tokenizer
        self.model = model
        self.tokenizer = tokenizer
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        self.messages = None
        self.hidden_states_layer_idx = hidden_states_layer_idx
        self.hidden_states_token_idx = hidden_states_token_idx

        # Set up generation parameters
        self.generation_kwargs = {
            "pad_token_id": self.tokenizer.eos_token_id,
            "output_hidden_states": True,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if additional_generation_kwargs:
            self.generation_kwargs.update(additional_generation_kwargs)
        self.additional_generation_kwargs_hook_fns = additional_generation_kwargs_hook_fns

        # Set up chat template parameters
        self.chat_template_kwargs = {
            "return_tensors": "pt",
            "add_generation_prompt": True,
            "return_dict": True,
        }
        if "qwen" in self.model_id.lower():
            self.chat_template_kwargs["enable_thinking"] = False

        # Steering parameters
        self.steering_kwargs, self.steering_hook = steering_kwargs, None
        if self.steering_kwargs is not None:
            self.steering_hook = SteeringHook(
                self.model,
                layer_index=self.steering_kwargs["layer_index"],
                token_index=self.steering_kwargs["token_index"],
                steering_vector=self.steering_kwargs["steering_vector"],
                steering_strength=self.steering_kwargs["steering_strength"],
            )

    def _get_response(self, messages, **kwargs):
        # return self._get_llm_response_and_hidden_states(messages=messages, **kwargs)
        return get_llm_response_and_hidden_states(
            messages=messages,
            model=self.model,
            tokenizer=self.tokenizer,
            chat_template_kwargs=self.chat_template_kwargs,
            generation_kwargs=self.generation_kwargs,
            additional_generation_kwargs_hook_fns=self.additional_generation_kwargs_hook_fns,
            hidden_state_token_idxs=self.hidden_states_token_idx,
            hidden_state_layer_idxs=self.hidden_states_layer_idx,
            **kwargs,
        )

    def clear_messages(self):
        """Clear the conversation messages."""
        self.messages = list()

    def get_llm_response_and_hidden_states(
        self,
        messages,
        **kwargs
    ):
        if self.steering_hook is None \
            or not self.steering_kwargs.get("selection_fn", lambda x: False)(locals()):
            return self._get_response(messages, **kwargs)
        else:
            with self.steering_hook:
                return self._get_response(messages, **kwargs)

    def add_message(self, utterance, send_with_next_message=True):
        """Add a message to the conversation."""
        if self.messages is None:
            self.messages = []
        self.messages.append({"role": "user", "content": utterance})
        if send_with_next_message:
            self.messages[-1]["add_to_next_message"] = True
        return self.messages

    def start_conversation(
        self,
        add_intro_message=True,
        send_to_assistant=True,
        save_logprobs=False,
        max_tokens=None,
    ):
        # Chat-style interaction using the specified model
        self.messages = []

        if add_intro_message:
            self.messages.append({"role": "user", "content": "Imagine that you are playing the board game Chameleon."})

        assistant_reply = None
        if send_to_assistant:
            assert add_intro_message, "You must add an intro message before sending to the assistant."
            llm_out = self.get_llm_response_and_hidden_states(
                self.messages,
                max_new_tokens=max_tokens,
                token_idx=self.hidden_states_token_idx,
                layer_idx=self.hidden_states_layer_idx,
                return_logprobs=save_logprobs,
            )
            self.messages.append({"role": "assistant", **llm_out})
            assistant_reply = llm_out["content"]
        elif add_intro_message:
            self.messages[-1]["add_to_next_message"] = True

        return assistant_reply, self.messages

    def send_message_get_response(
        self,
        utterance,
        max_tokens=None,
        **kwargs,
    ):
        if len(self.messages) > 0 \
            and self.messages[-1]["role"] == "user" \
            and self.messages[-1].get("add_to_next_message", False):

            # If the last message was from the user and we are continuing it
            self.messages[-1]["content"] = self.messages[-1]["content"] + "\n" + utterance
            self.messages[-1]["add_to_next_message"] = False
        else:
            self.messages.append({"role": "user", "content": utterance})

        llm_out = self.get_llm_response_and_hidden_states(
            self.messages,
            max_new_tokens=max_tokens,
            **kwargs,
        )
        self.messages.append({"role": "assistant", **llm_out})

        return llm_out["content"], self.messages
