import pyterrier as pt

def _attach_text(index):
    return pt.text.get_text(index, "text")


def create_bm25(index, num_results: int = 1000):
    bm25 = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=num_results
    )
    return bm25 >> _attach_text(index)


def create_bm25_rm3(
    index,
    num_results: int = 1000,
    fb_docs: int = 10,
    fb_terms: int = 10,
    fb_lambda: float = 0.6
):
    first_stage = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=num_results
    )

    rm3 = pt.rewrite.RM3(
        index,
        fb_docs=fb_docs,
        fb_terms=fb_terms,
        fb_lambda=fb_lambda
    )

    second_stage = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=num_results
    )

    return first_stage >> rm3 >> second_stage >> _attach_text(index)


def run_retrieval(retriever, queries):
    return retriever.transform(queries)
