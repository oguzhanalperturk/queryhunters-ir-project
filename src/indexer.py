import os
import shutil
import pyterrier as pt

from src.data_loader import iter_documents


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
    text_meta_length: int = 2048
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

    os.makedirs(index_path, exist_ok=True)

    print(f"Creating new Terrier index at: {index_path}")
    print(
        "Indexing the FULL passage corpus (8.8M passages). "
        "This is a one time step and may take a while; the corpus is "
        "downloaded automatically by ir_datasets on first use."
    )

    indexer = pt.IterDictIndexer(
        index_path,
        meta={
            "docno": 64,
            "text": text_meta_length,
        },
        text_attrs=["text"],
        fields=True,
        overwrite=True,
    )

    index_ref = indexer.index(iter_documents(dataset, max_docs=None))

    print("Index creation completed.")

    return pt.IndexFactory.of(index_ref)
