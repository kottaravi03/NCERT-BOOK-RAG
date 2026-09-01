"""
RAG Evaluation using Hybrid Retrieval + DeepEval
"""

import os
import uuid

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import (
    get_embeddings_from_gcp,
    get_model_from_gcp,
)

from deepeval import evaluate
from deepeval.evaluate import CacheConfig, AsyncConfig
from deepeval.models import GeminiModel

from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
)

from deepeval.test_case import LLMTestCase


load_dotenv()


# ============================================================
# 1. BUILD DOCUMENTS
# ============================================================

def build_documents() -> list[Document]:

    raw_chunks = [

        {
            "text": (
                "A fraction represents a part of a whole. "
                "When we divide a pizza into 4 equal slices "
                "and take 1, we have 1/4 of the pizza."
            ),
            "metadata": {
                "chapter": "Fractions",
                "page": 12,
            },
        },

        {
            "text": (
                "Dividing a cake into equal portions and taking "
                "one part is the same idea as a fraction."
            ),
            "metadata": {
                "chapter": "Fractions",
                "page": 13,
            },
        },

        {
            "text": (
                "Photosynthesis is the process by which plants "
                "convert sunlight into chemical energy."
            ),
            "metadata": {
                "chapter": "Biology",
                "page": 45,
            },
        },

        {
            "text": (
                "Exercise 4.3 asks students to calculate area "
                "of a triangle using the formula half base times height."
            ),
            "metadata": {
                "chapter": "Geometry",
                "page": 88,
            },
        },
    ]

    return [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        for chunk in raw_chunks
    ]


# ============================================================
# 2. DENSE RETRIEVER
# ============================================================

def build_dense_retriever(
    documents: list[Document],
) -> VectorStoreRetriever:

    embedding_model = get_embeddings_from_gcp()

    collection_name = (
        f"retriever_demo_{uuid.uuid4().hex[:8]}"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=collection_name,
    )

    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 2,
        },
    )

    return dense_retriever


# ============================================================
# 3. SPARSE RETRIEVER
# ============================================================

def build_sparse_retriever(
    documents: list[Document],
) -> BM25Retriever:

    sparse_retriever = BM25Retriever.from_documents(
        documents
    )

    sparse_retriever.k = 2

    return sparse_retriever


# ============================================================
# 4. HYBRID RETRIEVER
# ============================================================

def build_hybrid_retriever(
    dense_retriever,
    sparse_retriever,
) -> EnsembleRetriever:

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            dense_retriever,
            sparse_retriever,
        ],
        weights=[
            0.5,
            0.5,
        ],
    )

    return hybrid_retriever


# ============================================================
# 5. FORMAT DOCUMENTS
# ============================================================

def format_docs(
    retrieved_docs: list[Document],
) -> str:

    return "\n\n".join(
        (
            f"[Page {doc.metadata.get('page', '?')}] "
            f"{doc.page_content}"
        )
        for doc in retrieved_docs
    )


# ============================================================
# 6. BUILD RAG CHAIN
# ============================================================

def build_rag_chain(retriever):

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the context below.

If the answer is not in the context, say "I don't know".

Cite the page number when available.

Context:
{context}

Question:
{question}
"""
    )

    llm = get_model_from_gcp()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ============================================================
# 7. RUN RAG + BUILD DEEPEVAL TEST CASES
# ============================================================

def run_chain_and_build_cases(
    retriever,
    rag_chain,
    cases: list[dict],
) -> list[LLMTestCase]:

    llm_test_cases = []

    for case in cases:

        question = case["question"]

        retrieved_docs = retriever.invoke(
            question
        )

        contexts = [
            (
                f"[Page {doc.metadata.get('page', '?')}] "
                f"{doc.page_content}"
            )
            for doc in retrieved_docs
        ]

        answer = rag_chain.invoke(question)

        print("\n" + "=" * 80)
        print(f"Question: {question}")
        print(f"Answer  : {answer}")

        print("\nRetrieved Context:")
        for context in contexts:
            print(f"- {context}")

        llm_test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=case["expected"],
                retrieval_context=contexts,
            )
        )

    return llm_test_cases


# ============================================================
# 8. DEEPEVAL EVALUATION
# ============================================================

def evaluate_chain(
    llm_test_cases: list[LLMTestCase],
):

    project = os.getenv(
        "GOOGLE_CLOUD_PROJECT"
    )

    location = os.getenv(
        "GOOGLE_CLOUD_LOCATION"
    )

    if not project:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT is not configured."
        )

    if not location:
        raise ValueError(
            "GOOGLE_CLOUD_LOCATION is not configured."
        )

    judge = GeminiModel(
        model="gemini-2.5-flash-lite",
        use_vertexai=True,
        project=project,
        location=location,
        temperature=0,
    )

    metrics = [

        ContextualPrecisionMetric(
            threshold=0.7,
            model=judge,
        ),

        ContextualRecallMetric(
            threshold=0.7,
            model=judge,
        ),

        FaithfulnessMetric(
            threshold=0.7,
            model=judge,
        ),

        AnswerRelevancyMetric(
            threshold=0.7,
            model=judge,
        ),
    ]

    results = evaluate(
        test_cases=llm_test_cases,
        metrics=metrics,

        # Avoid Windows disk-cache problems
        cache_config=CacheConfig(
            use_cache=False,
            write_cache=False,
        ),

        # Easier to debug during development
        async_config=AsyncConfig(
            run_async=False,
        ),
    )

    return results


# ============================================================
# 9. MAIN
# ============================================================

def main():

    cases = [

        {
            "question": (
                "What does it mean to split something "
                "into equal parts?"
            ),
            "expected": (
                "A fraction is what you get when you "
                "divide something into equal parts."
            ),
        },

        {
            "question": (
                "What formula does the Exercise 4.3 use?"
            ),
            "expected": (
                "Exercise 4.3 uses the triangle area "
                "formula: half of the base times height."
            ),
        },
    ]

    # --------------------------------------------------------
    # Build documents
    # --------------------------------------------------------

    documents = build_documents()

    # --------------------------------------------------------
    # Build retrievers
    # --------------------------------------------------------

    dense_retriever = build_dense_retriever(
        documents
    )

    sparse_retriever = build_sparse_retriever(
        documents
    )

    hybrid_retriever = build_hybrid_retriever(
        dense_retriever,
        sparse_retriever,
    )

    # --------------------------------------------------------
    # Build RAG
    # --------------------------------------------------------

    rag_chain = build_rag_chain(
        hybrid_retriever
    )

    # --------------------------------------------------------
    # Create DeepEval test cases
    # --------------------------------------------------------

    llm_test_cases = (
        run_chain_and_build_cases(
            hybrid_retriever,
            rag_chain,
            cases,
        )
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate_chain(
        llm_test_cases
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("DeepEval Evaluation Results")
    print("=" * 100)

    for i, test_result in enumerate(
        results.test_results,
        start=1,
    ):

        print("\n" + "=" * 100)
        print(f"Test Case #{i}")
        print("=" * 100)

        print(
            f"Name     : {test_result.name}"
        )

        print(
            f"Question : {test_result.input}"
        )

        print(
            f"Answer   : {test_result.actual_output}"
        )

        print("\nMetrics")
        print("-" * 100)

        for metric in test_result.metrics_data:

            status = (
                "PASS"
                if metric.success
                else "FAIL"
            )

            print(
                f"Metric      : {metric.name}"
            )

            print(
                f"Score       : {metric.score:.3f}"
            )

            print(
                f"Threshold   : {metric.threshold}"
            )

            print(
                f"Status      : {status}"
            )

            print(
                f"Reason      : {metric.reason}"
            )

            print(
                f"Cost        : {metric.evaluation_cost}"
            )

            print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()