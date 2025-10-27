import requests

from task.models.message import Message
from task.models.role import Role


class ChatCompletionClient:
    _endpoint: str
    _api_key: str

    def __init__(self, model_name: str, api_key: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._endpoint = 'https://api.openai.com/v1/chat/completions'
        self._api_key = "Bearer " + api_key
        self._model_name = model_name

    def get_completion(
            self, messages: list[Message],
            print_request: bool = False,
            **kwargs
    ) -> Message:
        if print_request:
            print(f"Getting completion for `{self._get_messages_str(messages)}` \n\n ---And such parameters: {kwargs}---")

        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json"
        }
        request_data = {
            "model": self._model_name,
            "messages": [msg.to_dict() for msg in messages],
            **kwargs,
        }

        response = requests.post(url=self._endpoint, headers=headers, json=request_data, timeout=60)

        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content")
                return Message(Role.AI, content)
            raise ValueError("No Choice has been present in the response")
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

    def _get_messages_str(self, messages: list[Message]) -> str:
        return "--------\n\n".join(
            [f"---Role: {message.role.upper()}---\n💬 Message: {message.content}" for message in messages]
        )
