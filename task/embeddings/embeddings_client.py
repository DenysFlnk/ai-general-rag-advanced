import requests


class EmbeddingsClient:
    _endpoint: str
    _api_key: str

    def __init__(self, model_name: str, api_key: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._endpoint = "https://api.openai.com/v1/embeddings"
        self._api_key = "Bearer " + api_key
        self._model_name = model_name

    def get_embeddings(
        self, input: list[str], dimensions: int
    ) -> dict[int, list[float]]:
        headers = {"Authorization": self._api_key, "Content-Type": "application/json"}
        body = {"input": input, "model": self._model_name, "dimensions": dimensions}

        response = requests.post(url=self._endpoint, headers=headers, json=body)

        if response.status_code == 200:
            data = response.json()["data"]

            return {embedding["index"]: embedding["embedding"] for embedding in data}

        raise Exception(f"{response.text}")


# Hint:
# Request:
# curl https://api.openai.com/v1/embeddings \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#     "input": "Your text string goes here",
#     "model": "text-embedding-3-small",
#     "dimensions": 384
#   }'
#
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
