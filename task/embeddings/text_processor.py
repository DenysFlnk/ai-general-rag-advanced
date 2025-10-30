from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import EmbeddingsClient
from task.utils.text import chunk_text

INSERT_SQL = (
    "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)"
)
TRANCATE_SQL = "TRUNCATE TABLE vectors"
FILE_NAME = "microwave_manual.txt"


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: EmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config["host"],
            port=self.db_config["port"],
            database=self.db_config["database"],
            user=self.db_config["user"],
            password=self.db_config["password"],
        )

    def process_text_file(
        self,
        file_name: str,
        chunk_size: int = 150,
        overlap: int = 40,
        dimensions: int = 384,
        trancate_table: bool = True,
    ):
        if trancate_table:
            self._truncate_table()

        chunks = None

        with open(f"task/embeddings/{FILE_NAME}", mode="r") as file:
            chunks = chunk_text(
                text=file.read(), chunk_size=chunk_size, overlap=overlap
            )

        embeddings = self.embeddings_client.get_embeddings(
            input=chunks, dimensions=dimensions
        )

        with self._get_connection() as connection:
            with connection.cursor() as cursor:
                for index, embedding in embeddings.items():
                    embedding_str = f"[{','.join(map(str, embedding))}]"
                    cursor.execute(
                        INSERT_SQL, (FILE_NAME, chunks[index], embedding_str)
                    )

    def _truncate_table(self):
        with self._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(TRANCATE_SQL)

    def search(
        self,
        search_mode: SearchMode,
        request: str,
        top_k: int,
        min_score: float,
        dimensions: int = 384,
    ) -> list[str]:
        user_embedding = self.embeddings_client.get_embeddings(
            input=[request], dimensions=dimensions
        )
        vector_list_strings = map(str, user_embedding[0])
        embedding_str = f"[{','.join(vector_list_strings)}]"

        with self._get_connection() as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                method = (
                    "<->" if search_mode == SearchMode.EUCLIDIAN_DISTANCE else "<=>"
                )

                sql = f"""
                SELECT text, embedding {method} %s::vector as distance
                FROM vectors
                WHERE embedding {method} %s::vector <= %s
                ORDER BY distance
                LIMIT %s
                """.format(method=method)

                cursor.execute(sql, [embedding_str, embedding_str, min_score, top_k])

                rows = cursor.fetchall()

                results = []
                for row in rows:
                    results.append(row["text"])

                return results


# SELECT text, embedding <->  '[0.23, -0.45, 0.67, ..., 0.12]'::vector AS distance
# FROM microwave_data
# WHERE embedding <->  '[0.23, -0.45, 0.67, ..., 0.12]'::vector <= {score}
# ORDER BY distance
# LIMIT {top_k};
