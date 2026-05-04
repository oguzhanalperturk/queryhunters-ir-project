import os
import shutil
import pyterrier as pt

from src.data_loader import iter_subset_documents_for_queries


def init_pyterrier():
    if not pt.java.started():
        pt.java.init()


def is_valid_index(index_path: str) -> bool:
    required_files = [
        "data.properties",
        "data.lexicon.fsomapfile",
        "data.inverted.bf",
    ]

    return all(
        os.path.exists(os.path.join(index_path, file))
        for file in required_files
    )


def build_index(
    dataset,
    index_path: str,
    selected_queries,
    qrels,
    max_distractor_docs: int = 50000
):
    init_pyterrier()

    index_path = os.path.abspath(index_path)
    print(f"Resolved absolute index path: {index_path}")

    if is_valid_index(index_path):
        print(f"Valid index found: {index_path}")
        return pt.IndexFactory.of(index_path)

    if os.path.exists(index_path):
        print(f"Removing invalid/incomplete index: {index_path}")
        shutil.rmtree(index_path)

    parent_dir = os.path.dirname(index_path)
    os.makedirs(parent_dir, exist_ok=True)

    print(f"Creating new Terrier index at: {index_path}")

    terrier_index = pt.terrier.TerrierIndex(index_path)

    indexer = terrier_index.indexer(
        meta={
            "docno": 64,
            "text": 4096,
        }
    )

    indexer.index(
        iter_subset_documents_for_queries(
            dataset=dataset,
            selected_queries=selected_queries,
            qrels=qrels,
            max_distractor_docs=max_distractor_docs
        )
    )

    print("Index creation completed.")

    return pt.IndexFactory.of(index_path)