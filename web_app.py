import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import chromadb
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from starlette.staticfiles import StaticFiles

import app as rag


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "user_documents"
MANIFEST_PATH = UPLOAD_DIR / "documents.json"
STATIC_DIR = BASE_DIR / "static"
WEB_COLLECTION_NAME = "web_pdf_chunks"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SESSION_COOKIE_NAME = "pdf_chat_session"
SESSION_TTL_SECONDS = 12 * 60 * 60
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_FAILURES = 5
logger = logging.getLogger(__name__)
_lock = threading.RLock()
_login_lock = threading.Lock()
_login_failures = defaultdict(deque)
_documents = {}
_client = None
_collection = None
_embedder = None


def _save_documents():
    """Persist document state so uploaded PDFs remain available after restart."""
    temporary_path = MANIFEST_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(list(_documents.values()), indent=2), encoding="utf-8")
    temporary_path.replace(MANIFEST_PATH)


def _public_document(document):
    """Return UI-safe document details without exposing local filesystem paths."""
    return {key: document[key] for key in (
        "id", "filename", "status", "page_count", "chunk_count", "progress", "error"
    )}


def _get_collection():
    """Open the web app's persistent collection, separate from the CLI index."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(BASE_DIR / rag.CHROMA_PATH))
        _collection = _client.get_or_create_collection(name=WEB_COLLECTION_NAME)
    return _collection


def _load_documents():
    """Load saved uploads and mark any interrupted indexing operation clearly."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        loaded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Invalid document manifest: {MANIFEST_PATH}")
        for document in loaded:
            if document.get("status") == "indexing":
                document.update(
                    status="error",
                    error="Indexing stopped when the app closed. Remove and upload again.",
                )
            _documents[document["id"]] = document
        _save_documents()
    _get_collection()


@asynccontextmanager
async def lifespan(_application):
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    _validate_security_config()
    _load_documents()
    yield


app = FastAPI(
    title="Chat with your PDFs", lifespan=lifespan,
    docs_url=None, redoc_url=None, openapi_url=None,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QuestionRequest(BaseModel):
    question: str


class LoginRequest(BaseModel):
    password: str = Field(max_length=1024)


def _validate_security_config():
    """Require a strong app password and signing secret before serving requests."""
    if len(os.getenv("APP_PASSWORD", "")) < 12:
        raise RuntimeError("Set APP_PASSWORD in .env to a password at least 12 characters long.")
    if len(os.getenv("APP_SESSION_SECRET", "")) < 32:
        raise RuntimeError(
            "Set APP_SESSION_SECRET in .env to a random secret at least 32 characters long."
        )
    if os.getenv("APP_COOKIE_SECURE", "false").lower() not in {"true", "false"}:
        raise RuntimeError("APP_COOKIE_SECURE must be either true or false.")


def _session_signature(payload):
    secret = os.environ["APP_SESSION_SECRET"].encode("utf-8")
    return hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _create_session_token():
    expires_at = int(time.time()) + SESSION_TTL_SECONDS
    payload = f"{expires_at}.{secrets.token_urlsafe(24)}"
    return f"{payload}.{_session_signature(payload)}"


def _valid_session(token):
    """Verify the signed session and reject expired or malformed cookies."""
    if not token:
        return False
    try:
        expires_text, nonce, signature = token.split(".", 2)
        expires_at = int(expires_text)
        payload = f"{expires_text}.{nonce}"
        return (
            time.time() < expires_at <= time.time() + SESSION_TTL_SECONDS
            and hmac.compare_digest(signature, _session_signature(payload))
        )
    except (ValueError, KeyError):
        return False


def require_auth(request: Request):
    """Reject API requests unless the browser presents a valid signed session."""
    if not _valid_session(request.cookies.get(SESSION_COOKIE_NAME)):
        raise HTTPException(status_code=401, detail="Sign in to use your PDF library.")


def _require_same_origin(request: Request):
    """Require a same-origin browser request before accepting a state change."""
    origin = request.headers.get("origin")
    host = request.headers.get("host", "")
    if not origin or not host:
        raise HTTPException(status_code=403, detail="A same-origin request is required.")
    parsed_origin = urlsplit(origin)
    if (
        parsed_origin.scheme != request.url.scheme
        or parsed_origin.netloc.lower() != host.lower()
    ):
        raise HTTPException(status_code=403, detail="Cross-origin requests are not allowed.")


def _check_login_rate_limit(client_ip):
    now = time.monotonic()
    with _login_lock:
        failures = _login_failures[client_ip]
        while failures and now - failures[0] >= LOGIN_WINDOW_SECONDS:
            failures.popleft()
        if len(failures) >= LOGIN_MAX_FAILURES:
            raise HTTPException(
                status_code=429, detail="Too many sign-in attempts. Try again in 15 minutes."
            )


def _record_login_failure(client_ip):
    with _login_lock:
        _login_failures[client_ip].append(time.monotonic())


@app.get("/api/auth/status")
def auth_status(request: Request):
    """Tell the browser whether the current session is authenticated."""
    return {"authenticated": _valid_session(request.cookies.get(SESSION_COOKIE_NAME))}


@app.post("/api/auth/login")
def login(request: Request, credentials: LoginRequest, response: Response):
    """Authenticate the configured app password and issue a signed session cookie."""
    _require_same_origin(request)
    client_ip = request.client.host if request.client else "unknown"
    _check_login_rate_limit(client_ip)
    expected_password = os.getenv("APP_PASSWORD", "")
    if not expected_password or not hmac.compare_digest(credentials.password, expected_password):
        _record_login_failure(client_ip)
        raise HTTPException(status_code=401, detail="Incorrect password.")
    with _login_lock:
        _login_failures.pop(client_ip, None)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=_create_session_token(),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=os.getenv("APP_COOKIE_SECURE", "false").lower() == "true",
        samesite="strict",
        path="/",
    )
    return {"authenticated": True}


@app.post("/api/auth/logout", dependencies=[Depends(require_auth)])
def logout(request: Request, response: Response):
    """Clear the browser session after a same-origin logout request."""
    _require_same_origin(request)
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        httponly=True,
        secure=os.getenv("APP_COOKIE_SECURE", "false").lower() == "true",
        samesite="strict",
        path="/",
    )
    return {"authenticated": False}


def _index_document(document_id):
    """Extract one saved PDF and add its chunks to the local vector collection."""
    global _embedder
    with _lock:
        document = _documents.get(document_id)
        if document is None:
            return
        path = UPLOAD_DIR / document["stored_name"]

    try:
        reader = PdfReader(path)
        chunks, sources = [], []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = rag.clean_text(page.extract_text() or "")
            page_chunks = rag.chunk_text(page_text, rag.CHUNK_SIZE, rag.OVERLAP)
            chunks.extend(page_chunks)
            sources.extend([(document_id, document["filename"], page_number)] * len(page_chunks))
        if not chunks:
            raise ValueError("No readable text found. This PDF may contain scanned images.")

        with _lock:
            document.update(
                status="indexing", page_count=len(reader.pages),
                chunk_count=len(chunks), progress=0, error="",
            )
            _save_documents()
            if _embedder is None:
                _embedder = rag.SentenceTransformer(rag.EMBED_MODEL)

        collection = _get_collection()
        batch_size = 32
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            embeddings = _embedder.encode(batch).tolist()
            metadata = [
                {"document_id": doc_id, "source_file": filename,
                 "page_number": page, "chunk_number": offset + index}
                for index, (doc_id, filename, page) in enumerate(sources[offset:offset + batch_size])
            ]
            collection.add(
                ids=[f"{document_id}_chunk_{offset + index}" for index in range(len(batch))],
                documents=batch, metadatas=metadata, embeddings=embeddings,
            )
            with _lock:
                document = _documents.get(document_id)
                if document is None:
                    collection.delete(where={"document_id": document_id})
                    return
                document["progress"] = min(100, round((offset + len(batch)) / len(chunks) * 100))
                _save_documents()

        with _lock:
            document = _documents.get(document_id)
            if document is not None:
                document.update(status="ready", progress=100)
                _save_documents()
    except Exception as error:
        logger.exception("Could not index uploaded PDF %s", document_id)
        try:
            _get_collection().delete(where={"document_id": document_id})
        except Exception:
            logger.exception("Could not remove partial index for PDF %s", document_id)
        with _lock:
            document = _documents.get(document_id)
            if document is not None:
                document.update(status="error", error=str(error), progress=0)
                _save_documents()


@app.get("/")
def home():
    """Serve the local PDF chat interface."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/documents", dependencies=[Depends(require_auth)])
def list_documents():
    """List saved PDFs and their indexing status."""
    with _lock:
        return [_public_document(document) for document in _documents.values()]


@app.post("/api/documents")
async def upload_documents(
    background_tasks: BackgroundTasks,
    request: Request,
    files: list[UploadFile] = File(...),
    _authenticated: None = Depends(require_auth),
):
    """Save selected PDF uploads locally and queue their indexing."""
    _require_same_origin(request)
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one PDF.")
    filenames = [
        Path((uploaded.filename or "").replace("\\", "/")).name for uploaded in files
    ]
    for filename in filenames:
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{filename or 'File'} is not a PDF.")
    created, saved_paths = [], []
    try:
        for uploaded, original_name in zip(files, filenames):
            document_id = uuid.uuid4().hex
            stored_name = f"{document_id}.pdf"
            destination = UPLOAD_DIR / stored_name
            saved_paths.append(destination)
            size = 0
            try:
                with destination.open("wb") as saved_file:
                    while content := await uploaded.read(1024 * 1024):
                        size += len(content)
                        if size > MAX_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"{original_name} exceeds the 50 MB limit.",
                            )
                        saved_file.write(content)
            finally:
                await uploaded.close()
            document = {
                "id": document_id, "filename": original_name, "stored_name": stored_name,
                "status": "indexing", "page_count": 0, "chunk_count": 0,
                "progress": 0, "error": "",
            }
            with _lock:
                _documents[document_id] = document
                _save_documents()
            background_tasks.add_task(_index_document, document_id)
            created.append(_public_document(document))
    except Exception:
        with _lock:
            for document in created:
                _documents.pop(document["id"], None)
            for path in saved_paths:
                path.unlink(missing_ok=True)
            _save_documents()
        raise
    return created


@app.delete("/api/documents/{document_id}", dependencies=[Depends(require_auth)])
def remove_document(document_id: str, request: Request):
    """Remove a saved PDF and only its chunks from the local index."""
    _require_same_origin(request)
    with _lock:
        document = _documents.get(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="PDF not found.")
        if document["status"] == "indexing":
            raise HTTPException(status_code=409, detail="Wait for indexing to finish before removing.")
        _get_collection().delete(where={"document_id": document_id})
        (UPLOAD_DIR / document["stored_name"]).unlink(missing_ok=True)
        del _documents[document_id]
        _save_documents()
    return {"removed": document_id}


@app.get("/api/documents/{document_id}/file", dependencies=[Depends(require_auth)])
def open_document(document_id: str):
    """Open one uploaded PDF in the browser's built-in PDF viewer."""
    with _lock:
        document = _documents.get(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="PDF not found.")
        path = (UPLOAD_DIR / document["stored_name"]).resolve()
        if path.parent != UPLOAD_DIR.resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail="PDF file is unavailable.")
        return FileResponse(path, media_type="application/pdf", filename=document["filename"],
                            content_disposition_type="inline")


@app.post("/api/questions", dependencies=[Depends(require_auth)])
def ask_question(request: QuestionRequest, http_request: Request):
    """Retrieve supporting passages and ask Groq to answer from that context."""
    global _embedder
    _require_same_origin(http_request)
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Enter a question.")
    with _lock:
        ready = any(item["status"] == "ready" for item in _documents.values())
        if not ready:
            raise HTTPException(status_code=409, detail="Add a readable PDF and wait for indexing.")
        collection = _get_collection()
        if collection.count() == 0:
            raise HTTPException(status_code=409, detail="No indexed passages are available.")
        if _embedder is None:
            _embedder = rag.SentenceTransformer(rag.EMBED_MODEL)
        try:
            embedding = _embedder.encode([question]).tolist()
            result = collection.query(
                query_embeddings=embedding, n_results=min(rag.TOP_K, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            retrieved = [
                (text, meta["chunk_number"], meta["document_id"], meta["source_file"],
                 meta["page_number"], distance)
                for text, meta, distance in zip(
                    result["documents"][0], result["metadatas"][0], result["distances"][0]
                )
            ]
            prompt = rag.build_prompt(
                question,
                [(text, number, filename, page, distance)
                 for text, number, _doc_id, filename, page, distance in retrieved],
            )
            answer = rag.ask_llm(prompt)
        except Exception as error:
            logger.exception("Question answering failed")
            raise HTTPException(status_code=502, detail=f"Could not get an answer: {error}") from error
    return {
        "answer": answer,
        "sources": [
            {"document_id": doc_id, "filename": filename, "page": page,
             "chunk": number, "text": text}
            for text, number, doc_id, filename, page, _distance in retrieved
        ],
    }
