import pandas as pd
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name)

    def rerank(self, results: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
        query_map = dict(zip(queries["qid"].astype(str), queries["query"]))

        rows = []
        pairs = []

        for _, row in results.iterrows():
            qid = str(row["qid"])
            query = query_map[qid]
            passage = row["text"]

            rows.append(row)
            pairs.append((query, passage))

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=True
        )

        reranked_rows = []

        for row, score in zip(rows, scores):
            new_row = row.copy()
            new_row["score"] = float(score)
            reranked_rows.append(new_row)

        reranked = pd.DataFrame(reranked_rows)

        reranked = reranked.sort_values(
            by=["qid", "score"],
            ascending=[True, False]
        )

        reranked["rank"] = reranked.groupby("qid").cumcount()

        return reranked