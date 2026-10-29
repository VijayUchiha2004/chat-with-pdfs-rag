# Chat with PDFs

A small local Retrieval-Augmented Generation (RAG) learning project with
terminal and browser interfaces. It reads PDF text, splits it into overlapping
chunks, embeds the chunks locally, retrieves relevant passages, and asks Groq
to answer using those passages.
It uses Python, pypdf, sentence-transformers, ChromaDB, the Groq SDK, and
python-dotenv directly; it does not use a RAG framework.

## What you need

- Python 3.10 or newer.
- Internet access to install packages, download the embedding model the first
  time it runs, and send questions to Groq.
- A Groq API key.
- One or more text-based PDF files. PDFs from any subject are supported.

Scanned/image-only PDFs need OCR, which this project does not include.

## Project files

```text
p1-chat-pdf/
├── app.py              # PDF extraction, cleaning, chunking, search, and answers
├── requirements.txt    # Python packages used by the app
├── .env                # Your local Groq API key; create this yourself, never share it
├── web_app.py          # Local API and browser interface backend
├── static/             # Browser interface files
├── data/               # Put the PDFs you want to search here
├── user_documents/     # Uploaded PDFs and local document list, created automatically
└── chroma_db/          # Local vector index, created automatically
```

Keep `.env` private. It is ignored by Git, along with `chroma_db/` and the
virtual environment.

## Setup on Windows PowerShell

Run these commands from the project folder:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
notepad .env
```

If `.env` does not exist, Notepad will ask to create it. Add your Groq key in
this format, save the file, and do not include quotes:

```env
GROQ_API_KEY=your_groq_api_key
```

The `.env` file is ignored by Git and must never be uploaded or shared.

## Add PDFs and run

Copy your PDFs into `data\`. The app indexes every `.pdf` file directly inside
the folder (it does not search nested folders by default). Then run:

```powershell
python app.py data
```

You can also index one PDF by giving its path:

```powershell
python app.py data\ods-cpp-1.pdf
```

Use `--recursive` to include PDFs inside nested folders. Chunking and retrieval
settings can be changed from the command line:

```powershell
python app.py data --recursive --chunk-size 700 --overlap 120 --top-k 5
```

The app prints extracted text previews and retrieved passages under `DEBUG`
before each answer. Citations show the PDF filename, page number, and chunk
number. Answers are prompted to use simple language and concise explanations;
the browser interface formats Markdown headings and lists for readability.
Type a question at the prompt; type `quit` to exit.

## What happens when it runs

1. Extract and clean text from each PDF page.
2. Split each page into 500-character chunks with 100 characters of overlap.
3. Generate embeddings with `all-MiniLM-L6-v2` and save them in `chroma_db\`.
4. Retrieve the three closest chunks for each question.
5. Send the question and retrieved context to Groq model `openai/gpt-oss-120b`.

The index is rebuilt from the selected PDF file or folder every time the app
starts. Each run replaces the previous Chroma collection; PDFs outside the
selected input are not included in that run.

## Troubleshooting

- **`ModuleNotFoundError`**: Activate the project venv, then run
  `python -m pip install -r requirements.txt`. Using `python -m pip` helps
  install packages into the same Python environment that runs the app.
- **Missing API key**: Check that `.env` is in the project folder and contains
  `GROQ_API_KEY=your_actual_key`.
- **No PDFs found**: Confirm the folder path and that the PDFs are directly
  inside it, not only in a nested folder.
- **No or poor answer**: The app can only use text it successfully extracts
  and retrieves. Scanned pages require OCR, and PDF layout or extraction
  quality can affect results. Check the printed preview and debug chunks.
- **Slow first run**: The embedding model is downloaded and loaded locally the
  first time; later runs can reuse the downloaded model.

## Use the browser interface

Install the requirements as above, add your Groq key to `.env`, then start the
local server:

```powershell
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser. Add PDFs using the upload button
or drag and drop them. Wait for each document to show **Ready**, ask a
question, and select a source card to inspect its passage or open the cited
PDF page. The interface accepts PDF files up to 50 MB each.

Uploads are stored under `user_documents\`, and the browser interface uses its
own persistent Chroma collection under `chroma_db\`. Documents remain available
after restarting the server. Use the remove button on a document card to delete
that uploaded copy and its indexed chunks. The command-line app keeps its
separate collection and continues to work independently.

The server binds to `127.0.0.1` so it is only available on this computer by
default. Do not expose it to a public network: this learning app has no login.
PDF files and embeddings stay local; each question and its retrieved passages
are sent to Groq to generate an answer.

## Run offline unit tests

The standard-library tests check text cleaning, chunking, CLI settings, and
source metadata without calling Groq or downloading an embedding model:

```powershell
python -m unittest discover -s tests -v
```

## Resume summary

> Built a local multi-PDF RAG application with a Python CLI and browser UI,
> using local sentence-transformer embeddings, persistent Chroma vector search,
> and Groq LLM responses with filename/page citations. Added asynchronous PDF
> uploads, indexing status, configurable ingestion, and offline API/pipeline tests.
