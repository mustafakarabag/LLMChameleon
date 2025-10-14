class NaturalLanguageTalker:
    """Abstract class to handle conversations via language models and human players."""

    def __init__(self, model_id=None, start_conversation=True):
        self.model_id = model_id
        if start_conversation:
            self.start_conversation()


    def start_conversation(self):
        raise NotImplementedError("start_conversation() must be implemented by subclass")

    def send_message_get_response(self, messages, max_tokens=10):
        raise NotImplementedError("send_message() must be implemented by subclass")
