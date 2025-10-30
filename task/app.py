import os

from task.chat.chat_completion_client import ChatCompletionClient
from task.embeddings.embeddings_client import EmbeddingsClient
from task.embeddings.text_processor import SearchMode, TextProcessor
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""

DB_CONFIG = {
    "host": "localhost",
    "port": 54333,
    "database": "vectordb",
    "user": "postgres",
    "password": "postgres",
}

api_key = os.getenv("OPENAI_API_KEY", "")
embeddings_client = EmbeddingsClient(
    model_name="text-embedding-3-small", api_key=api_key
)
chat_completion_client = ChatCompletionClient(model_name="gpt-4o", api_key=api_key)
text_processor = TextProcessor(embeddings_client=embeddings_client, db_config=DB_CONFIG)


def main():
    text_processor.process_text_file(file_name="microwave_manual.txt")
    conversation = Conversation(messages=[Message(Role.SYSTEM, SYSTEM_PROMPT)])

    while True:
        user_input = input("🧒 -> ")

        if user_input.lower() == "exit":
            print("Chat ended 🔚")
            print("=" * 100)
            exit(0)

        rag_context = text_processor.search(
            request=user_input,
            top_k=5,
            min_score=0.5,
            search_mode=SearchMode.COSINE_DISTANCE,
        )
        augmented_request = USER_PROMPT.format(
            context="\n".join(rag_context), query=user_input
        )

        conversation.add_message(Message(Role.USER, augmented_request))

        response = chat_completion_client.get_completion(
            messages=conversation.get_messages()
        )
        conversation.add_message(response)

        print(f"🤖 -> {response.content}")


main()
