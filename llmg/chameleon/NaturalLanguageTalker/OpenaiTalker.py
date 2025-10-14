from llmg.chameleon.NaturalLanguageTalker.naturallanguagetalker import NaturalLanguageTalker
from openai import OpenAI


class OpenaiTalker(NaturalLanguageTalker):
    """Class to handle conversations for OpenAI players."""

    def __init__(
        self,
        model_id,
        api_key=None,
        additional_generation_kwargs=None,
        start_conversation=False,
    ):
        self.messages = None
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.generation_kwargs = {
            "model": model_id,
        }
        if additional_generation_kwargs:
            self.generation_kwargs.update(additional_generation_kwargs)

        super().__init__(model_id=model_id, start_conversation=start_conversation)

    def _get_response(self, messages, max_tokens=None, **kwargs):
        # Prepare generation options
        gen_kwargs = self.generation_kwargs.copy() if self.generation_kwargs else dict()
        if max_tokens is not None:
            gen_kwargs["max_output_tokens"] = max_tokens
        gen_kwargs.update(kwargs)
        
        # Generate response
        input_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages] # Remove additional fields
        response = self.client.responses.create(input=input_messages, **gen_kwargs) # Newer Responses API
        
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
            self.messages.append({"role": "system", "content": "Imagine that you are playing the board game Chameleon."})

        assistant_reply = None
        if send_to_assistant:
            assert add_intro_message, "You must add an intro message before sending to the assistant."
            assistant_reply = self._get_response(self.messages, max_tokens=max_tokens).choices[0].message.content
            self.messages.append({"role": "assistant", "content": assistant_reply})
        elif add_intro_message:
            self.messages[-1]["add_to_next_message"] = True

        return assistant_reply, self.messages

    def send_message_get_response(
        self,
        utterance,
        max_tokens=None,
    ):
        # Prepare message for the assistant
        if len(self.messages) > 0 \
            and self.messages[-1]["role"] != "assistant" \
            and self.messages[-1].get("add_to_next_message", False):

            # If the last message was from the user and we are continuing it
            self.messages[-1]["content"] = self.messages[-1]["content"] + "\n" + utterance
            self.messages[-1]["add_to_next_message"] = False
        else:
            self.messages.append({"role": "user", "content": utterance})

        # Get response from the assistant
        assistant_api_response = self._get_response(self.messages, max_tokens=max_tokens)
        assistant_reply = assistant_api_response.output_text # Newer Responses API
        self.messages.append({
            "role": "assistant",
            "content": assistant_reply,
            "total_tokens": assistant_api_response.usage.total_tokens,
            "input_tokens": assistant_api_response.usage.input_tokens,
            "output_tokens": assistant_api_response.usage.output_tokens,
            "reasoning_tokens": getattr(getattr(assistant_api_response.usage, "output_tokens_details", 0), "reasoning_tokens", 0),
        })

        return assistant_reply, self.messages
