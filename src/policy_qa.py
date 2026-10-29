
from pathlib import Path
import csv
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm

POLICY_FOLDER = PROJECT_ROOT / "data" / "policies"
EVAL_FILE = PROJECT_ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

FALLBACK = (
    "I could not find sufficient information in the policy documents "
    "to answer that question."
)

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10


def load_policy_documents():
    """Load policy documents and preserve each source filename."""
    documents = []

    for path in sorted(POLICY_FOLDER.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        documents.append({
            "source": path.name,
            "text": text
        })

    if not documents:
        raise FileNotFoundError(
            f"No policy .txt files found in {POLICY_FOLDER}"
        )

    return documents


def chunk_documents(documents, overlap=CHUNK_OVERLAP):
    """Split documents into 40-word chunks with the requested overlap."""
    chunks = []
    step = CHUNK_SIZE - overlap

    if step <= 0:
        raise ValueError("Overlap must be smaller than chunk size.")

    for document in documents:
        words = document["text"].split()
        start = 0

        while start < len(words):
            chunk_words = words[start:start + CHUNK_SIZE]

            if not chunk_words:
                break

            chunks.append({
                "source": document["source"],
                "text": " ".join(chunk_words)
            })

            if start + CHUNK_SIZE >= len(words):
                break

            start += step

    return chunks


class PolicyRetriever:
    """TF-IDF and cosine-similarity search over policy chunks."""

    def __init__(self, chunks):
        if not chunks:
            raise ValueError("Cannot build a retriever with no chunks.")

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in chunks]
        )

    def retrieve(self, query, k=3):
        """Return the k most relevant chunks with source and score."""
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()

        ranked_indices = scores.argsort()[::-1][:k]

        results = []
        for index in ranked_indices:
            results.append({
                "source": self.chunks[index]["source"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"]
            })

        return results


def answer(question, retriever):
    """Answer using retrieved policy evidence and return source filenames."""
    results = retriever.retrieve(question, k=3)

    # If the query has no matching vocabulary, do not invent an answer.
    if not results or max(item["score"] for item in results) == 0:
        return FALLBACK, []

    context = "\n".join(
        f"[{item['source']}] {item['text']}"
        for item in results
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a customer support POLICY ASSISTANT. "
                "Answer using only the supplied policy context. "
                "Do not invent rules, amounts, deadlines, or exceptions. "
                "If the context does not contain enough information to answer, "
                f"reply exactly with this fallback: {FALLBACK}"
            )
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n"
                f"QUESTION:\n{question}"
            )
        }
    ]

    response = mock_llm.chat(messages)
    answer_text = response.choices[0].message.content.strip()
    sources = sorted({item["source"] for item in results})

    return answer_text, sources


def is_unanswerable(row):
    """Identify evaluation rows without an expected document or fact."""
    doc = row.get("expected_doc", "").strip().lower()
    fact = row.get("expected_fact", "").strip().lower()

    return (
        not doc
        or doc in {"none", "unanswerable", "unknown", "n/a"}
        or not fact
        or fact in {"none", "unanswerable", "unknown", "n/a"}
    )


def evaluate_retrieval_and_answers(retriever, evaluation_rows):
    """Evaluate top-three document retrieval, expected facts, and fallback."""
    results = []

    for row in evaluation_rows:
        question = row["question"].strip()
        expected_doc = row.get("expected_doc", "").strip()
        expected_fact = row.get("expected_fact", "").strip()
        unanswerable = is_unanswerable(row)

        retrieved = retriever.retrieve(question, k=3)
        retrieved_sources = [item["source"] for item in retrieved]

        response_text, answer_sources = answer(question, retriever)

        expected_doc_found = (
            not unanswerable and expected_doc in retrieved_sources
        )
        expected_fact_found = (
            not unanswerable and expected_fact in response_text
        )
        fallback_correct = (
            unanswerable
            and response_text.startswith("I could not find")
        )

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "retrieved_docs": ";".join(retrieved_sources),
            "answer_sources": ";".join(answer_sources),
            "answer": response_text,
            "is_unanswerable": unanswerable,
            "expected_doc_found_top3": expected_doc_found,
            "expected_fact_found": expected_fact_found,
            "fallback_correct": fallback_correct
        })

    return results


def save_evaluation(results, filename):
    """Save individual evaluation results and print the score summary."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_FOLDER / filename

    if not results:
        raise ValueError("The policy evaluation file has no rows.")

    with output_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    answerable = [r for r in results if not r["is_unanswerable"]]
    unanswerable = [r for r in results if r["is_unanswerable"]]

    doc_correct = sum(r["expected_doc_found_top3"] for r in answerable)
    fact_correct = sum(r["expected_fact_found"] for r in answerable)
    fallback_correct = sum(r["fallback_correct"] for r in unanswerable)

    print(f"\nSaved evaluation details to: {output_path}")
    print(f"Total evaluation questions: {len(results)}")
    print(f"Answerable questions: {len(answerable)}")
    print(f"Unanswerable questions: {len(unanswerable)}")
    print(
        f"Expected document found in top 3: "
        f"{doc_correct}/{len(answerable)}"
    )
    print(
        f"Expected fact found in answer: "
        f"{fact_correct}/{len(answerable)}"
    )
    print(
        f"Unanswerable questions with correct fallback: "
        f"{fallback_correct}/{len(unanswerable)}"
    )


def main():
    print("Instamart Policy QA — TF-IDF Retrieval")
    print("=" * 42)

    documents = load_policy_documents()
    chunks = chunk_documents(documents, overlap=CHUNK_OVERLAP)

    print(f"Policy documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} words")
    print(f"Overlap: {CHUNK_OVERLAP} words")

    for source in sorted({chunk["source"] for chunk in chunks}):
        count = sum(chunk["source"] == source for chunk in chunks)
        print(f"  {source}: {count} chunks")

    retriever = PolicyRetriever(chunks)

    # Required retrieval sanity check.
    print("\nTop 3 results for: refund policy late delivery")
    for item in retriever.retrieve("refund policy late delivery", k=3):
        print(
            f"  {item['source']} | score={item['score']:.4f} | "
            f"{item['text'][:100]}..."
        )

    with EVAL_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        evaluation_rows = list(csv.DictReader(file))

    if not evaluation_rows:
        raise ValueError(f"No evaluation rows found in {EVAL_FILE}")

    print("\nRunning policy evaluation...")
    results = evaluate_retrieval_and_answers(
        retriever, evaluation_rows
    )
    save_evaluation(results, "policy_eval_results.csv")

    # Repeat evaluation with zero overlap to compare chunk boundaries.
    zero_overlap_chunks = chunk_documents(documents, overlap=0)
    zero_overlap_retriever = PolicyRetriever(zero_overlap_chunks)

    print("\nRunning zero-overlap evaluation...")
    zero_results = evaluate_retrieval_and_answers(
        zero_overlap_retriever, evaluation_rows
    )
    save_evaluation(zero_results, "policy_eval_zero_overlap.csv")

    # Interactive questions after evaluation.
    print("\nYou can now ask policy questions.")
    print("Type 'exit' to finish.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            print("Please enter a question.")
            continue

        response_text, sources = answer(question, retriever)

        print("\nAnswer:")
        print(response_text)
        print("\nSources:")
        for source in sources:
            print(f"- {source}")

        if not sources:
            print("- No relevant policy source found.")


if __name__ == "__main__":
    main()
