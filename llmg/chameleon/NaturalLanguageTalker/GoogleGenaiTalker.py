import os
from llmg.chameleon.NaturalLanguageTalker.naturallanguagetalker import NaturalLanguageTalker
from google import genai


class GoogleGenaiTalker(NaturalLanguageTalker):
    """Class to handle conversations for Google GenAI players."""

    def __init__(
        self,
        model_id,
        api_key=None,
        additional_generation_kwargs=None,
        start_conversation=False,
    ):
        self.client = genai.Client(api_key=api_key)
        self.generation_kwargs = {
            "safety_settings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
                                {
                    "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
                    "threshold": "BLOCK_NONE",
                },
            ]
        }
        if additional_generation_kwargs:
            self.generation_kwargs.update(additional_generation_kwargs)

        super().__init__(model_id=model_id, start_conversation=start_conversation)
        self.messages = None

    def _get_response(self, messages, max_tokens=None, **kwargs):
        # Prepare generation options
        gen_kwargs = self.generation_kwargs.copy() if self.generation_kwargs else dict()
        if max_tokens is not None:
            gen_kwargs["max_output_tokens"] = max_tokens
        gen_kwargs.update(kwargs)

        # Generate response
        input_messages = [{
            "role": {"user": "user", "assistant": "model"}[msg["role"]],
            "parts": [{"text": msg["content"]}]
        } for msg in messages] # Remove additional fields and remap to google genai format
        chat = self.client.chats.create(model=self.model_id, config=gen_kwargs, history=input_messages[:-1])  # Exclude the last message to avoid repetition
        response = chat.send_message(input_messages[-1]["parts"][0]["text"])

        return response

    def clear_messages(self):
        """Clear the conversation history."""
        self.messages = []

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
        max_tokens=None,
    ):
        # Chat-style interaction using the specified model
        self.messages = []

        if add_intro_message:
            self.messages.append({"role": "user", "content": "Imagine that you are playing the board game Chameleon."})

        assistant_reply = None
        if send_to_assistant:
            assert add_intro_message, "You must add an intro message before sending to the assistant."
            assistant_reply = self._get_response(self.messages, max_tokens=max_tokens).text
            self.messages.append({"role": "assistant", "content": assistant_reply})
        elif add_intro_message:
            self.messages[-1]["add_to_next_message"] = True

        return assistant_reply, self.messages

    def send_message_get_response(
        self,
        utterance,
        max_tokens=None,
    ):
        # Prepare message for the model
        if len(self.messages) > 0 \
            and self.messages[-1]["role"] != "assistant" \
            and self.messages[-1].get("add_to_next_message", False):

            # If the last message was from the user and we are continuing it
            self.messages[-1]["content"] = self.messages[-1]["content"] + "\n" + utterance
            self.messages[-1]["add_to_next_message"] = False
        else:
            self.messages.append({"role": "user", "content": utterance})

        # Get response from the model
        assistant_api_response = self._get_response(self.messages, max_tokens=max_tokens)
        assistant_reply = assistant_api_response.text
        if assistant_reply is None:
            print(f"[WARNING] Assistant reply is None.")
            assistant_reply = ""
        self.messages.append({
            "role": "assistant",
            "content": assistant_reply,
            "input_tokens": assistant_api_response.usage_metadata.prompt_token_count,
            "output_tokens": assistant_api_response.usage_metadata.candidates_token_count,
            "reasoning_tokens": assistant_api_response.usage_metadata.thoughts_token_count,
        })

        return assistant_reply, self.messages
