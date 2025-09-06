import json

import requests

#TODO:
# ---
# https://platform.openai.com/docs/api-reference/embeddings
# ---
# Implement EmbeddingsClient:
# - constructor should apply model name and api key
# - endpoint is https://api.openai.com/v1/embeddings
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

class EmbeddingsClient:
    _endpoint: str
    _api_key: str

    def __init__(self, model_name: str, api_key: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._endpoint = 'https://api.openai.com/v1/embeddings'
        self._api_key = "Bearer " + api_key
        self._model_name = model_name

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

