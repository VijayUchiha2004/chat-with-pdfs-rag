import argparse
import os
import re
import sys
import unicodedata
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# These constants keep the main RAG settings easy to find and adjust.
CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 3
MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_chunks"
CHROMA_PATH = "./chroma_db"

# Reuse the local embedder and indexed collection across questions.
_embedder = _collection = None
_page_texts = []


# Extracting text turns the PDF into material that can be searched and answered.
def extract_text(pdf_path) -> str:
    """Read every PDF page and preview extracted text to check extraction quality."""
    global _page_texts
    reader = PdfReader(pdf_path)
    _page_texts = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(_page_texts)
    print(f"Pages: {len(reader.pages)}")
    print("Text preview (first 300 characters):")
    print(text[:300])
    return text

# Cleaning fixes PDF ligatures and line breaks before they affect chunking.
def clean_text(text) -> str:
    """Normalize Unicode and whitespace so chunks contain readable paragraphs."""
    text = unicodedata.normalize("NFKC", text).replace("-\n", "")
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r" {2,}", " ", text)

# Overlapping character windows keep nearby context from being cut at chunk edges.
def chunk_text(text, chunk_size=500, overlap=100) -> list[str]:
    """Split text into overlapping character chunks so retrieval keeps local context."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than it.")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks

# Embeddings let Chroma find chunks by meaning rather than exact word matching.
def build_index(chunks, sources):
    """Embed and store chunks, replacing the old collection to prevent stale results."""
    global _embedder, _collection
    if len(chunks) != len(sources):
        raise ValueError("Each chunk must have a matching PDF and page number.")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection_names = [item if isinstance(item, str) else item.name for item in client.list_collections()]
    if COLLECTION_NAME in collection_names:
        client.delete_collection(name=COLLECTION_NAME)
    _collection = client.create_collection(name=COLLECTION_NAME)
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = _embedder.encode(chunks).tolist()
    _collection.add(ids=[f"chunk_{number}" for number in range(len(chunks))],
                    documents=chunks,
                    metadatas=[{"chunk_number": number, "source_file": filename,
                                "page_number": page}
                               for number, (filename, page) in enumerate(sources)],
                    embeddings=embeddings)

# A question embedding finds the closest stored passages that may contain its answer.
def retrieve(question, k=3) -> list[tuple[str, int, str, int, float]]:
    """Return closest chunks with their PDF filenames, page numbers, and distances."""
    if _collection is None:
        raise RuntimeError("The PDF index has not been built.")
    if _embedder is None:
        raise RuntimeError("The embedding model has not been loaded.")
    question_embedding = _embedder.encode([question]).tolist()
    results = _collection.query(
        query_embeddings=question_embedding, n_results=min(k, _collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    return [
        (document, metadata["chunk_number"], metadata["source_file"],
         metadata["page_number"], distance)
        for document, metadata, distance
        in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]

# The retrieved passages ground the model's answer in the source PDF.
def build_prompt(question, retrieved_chunks) -> str:
    """Combine the question and retrieved context while requiring a document-only answer."""
    context = "\n\n".join(
        f"[{filename}, page {page}, chunk {number}]\n{text}"
        for text, number, filename, page, _ in retrieved_chunks
    )
    return (
        "Answer using ONLY the context below.\n"
        "If the answer is not in the context, reply exactly:\n"
        "'I could not find this in the document.'\n\n"
        "Write for a learner who is new to the topic. Start with a direct answer. "
        "Use simple words, short sentences, and short paragraphs. Explain technical "
        "terms briefly when needed. Use a short list only when it makes steps or "
        "multiple points clearer. Avoid repetition and unrelated detail. "
        "Usually answer in 3 to 6 sentences; use more only when the question needs it. "
        "Use light Markdown formatting for headings, lists, and important terms.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

# Groq generates a natural-language answer from the grounded prompt.
def ask_llm(prompt) -> str:
    """Send the retrieval-grounded prompt to Groq and return its answer."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing; add it to your .env file.")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(model=MODEL,
        messages=[{"role": "user", "content": prompt}], temperature=0.1)
    answer = response.choices[0].message.content
    if answer is None:
        raise RuntimeError("Groq returned an empty response.")
    return answer

# This command-line loop indexes one PDF, then answers questions until the user quits.
# Command-line options make corpus selection and retrieval tuning repeatable.
def parse_args(argv=None):
    """Parse the input path and RAG settings supplied by the user."""
    parser = argparse.ArgumentParser(description="Ask questions about one or more PDFs.")
    parser.add_argument("source", type=Path, help="PDF file or folder containing PDFs")
    parser.add_argument("--recursive", action="store_true",
                        help="include PDFs in nested folders")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"chunk length in characters (default: {CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=OVERLAP,
                        help=f"overlap in characters (default: {OVERLAP})")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"number of chunks to retrieve (default: {TOP_K})")
    args = parser.parse_args(argv)
    if args.chunk_size <= 0 or not 0 <= args.overlap < args.chunk_size:
        parser.error("--chunk-size must be positive and --overlap smaller than it")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


# This command-line loop indexes the chosen PDFs and answers questions until quit.
def main(argv=None):
    """Run the PDF-to-answer pipeline from the terminal."""
    args = parse_args(argv)
    input_path = args.source
    if input_path.is_dir():
        candidates = input_path.rglob("*") if args.recursive else input_path.iterdir()
        pdf_paths = sorted(
            (path for path in candidates
             if path.is_file() and path.suffix.lower() == ".pdf"),
            key=lambda path: str(path).lower(),
        )
        if not pdf_paths:
            print(f"Error: no PDF files found in folder: {input_path}")
            return
    elif input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdf_paths = [input_path]
    else:
        print(f"Error: PDF file or folder not found: {input_path}")
        return
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY is missing; copy .env.example to .env and add your key.")
        return
    chunks, sources = [], []
    for pdf_path in pdf_paths:
        source_name = (pdf_path.relative_to(input_path).as_posix()
                       if input_path.is_dir() else pdf_path.name)
        text = clean_text(extract_text(pdf_path))
        if not text.strip():
            print(f"Error: no text could be extracted from {pdf_path.name}.")
            return
        for page_number, page_text in enumerate(_page_texts, start=1):
            page_chunks = chunk_text(clean_text(page_text), args.chunk_size, args.overlap)
            chunks.extend(page_chunks)
            sources.extend([(source_name, page_number)] * len(page_chunks))
    if not chunks:
        print("Error: the PDF did not produce any text chunks.")
        return
    print(f"\nIndexing {len(chunks)} chunks...")
    build_index(chunks, sources)
    print("Ready. Ask questions about the PDF.")
    while True:
        question = input("\nQuestion (or 'quit'): ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        retrieved_chunks = retrieve(question, args.top_k)
        print("\nDEBUG — retrieved chunks:")
        for chunk, number, filename, page, distance in retrieved_chunks:
            print(f"\n{filename}, page {page}, chunk {number} "
                  f"(distance {distance:.4f}):\n{chunk}")
        prompt = build_prompt(question, retrieved_chunks)
        print("\nAnswer:")
        print(ask_llm(prompt))
        sources = ", ".join(
            f"{filename}, page {page} (chunk {number})"
            for _, number, filename, page, _ in retrieved_chunks
        )
        print(f"Sources: {sources}")

if __name__ == "__main__":
    main()
