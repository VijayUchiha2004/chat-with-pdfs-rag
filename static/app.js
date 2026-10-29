const documentList = document.querySelector("#document-list");
const documentCount = document.querySelector("#document-count");
const dropZone = document.querySelector("#drop-zone");
const fileInput = document.querySelector("#file-input");
const addButton = document.querySelector("#add-button");
const addMoreButton = document.querySelector("#add-more-button");
const chatContent = document.querySelector("#chat-content");
const questionForm = document.querySelector("#question-form");
const questionInput = document.querySelector("#question-input");
const askButton = document.querySelector("#ask-button");
const globalError = document.querySelector("#global-error");
const sourcePanel = document.querySelector("#source-panel");
const authScreen = document.querySelector("#auth-screen");
const appShell = document.querySelector("#app-shell");
const loginForm = document.querySelector("#login-form");
const passwordInput = document.querySelector("#password-input");
const loginError = document.querySelector("#login-error");
const logoutButton = document.querySelector("#logout-button");

let documents = [];
let selectedSourceButton = null;
let isAsking = false;
let documentPoller = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function formatInline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
}

function formatAnswer(answer) {
  const output = [];
  let paragraph = [];
  let listType = null;
  const closeList = () => {
    if (listType) output.push(`</${listType}>`);
    listType = null;
  };
  const flushParagraph = () => {
    if (paragraph.length) output.push(`<p>${paragraph.map(formatInline).join(" ")}</p>`);
    paragraph = [];
  };

  for (const line of String(answer).split(/\r?\n/)) {
    const heading = line.match(/^#{1,3}\s+(.+)$/);
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (!line.trim()) {
      flushParagraph();
      closeList();
    } else if (heading) {
      flushParagraph();
      closeList();
      output.push(`<h3>${formatInline(heading[1])}</h3>`);
    } else if (bullet || numbered) {
      flushParagraph();
      const nextType = bullet ? "ul" : "ol";
      if (listType !== nextType) {
        closeList();
        output.push(`<${nextType}>`);
        listType = nextType;
      }
      output.push(`<li>${formatInline((bullet || numbered)[1])}</li>`);
    } else {
      closeList();
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
  closeList();
  return output.join("");
}

function showError(message) {
  globalError.innerHTML = `<strong>Could not complete that action.</strong><br>${escapeHtml(message)}`;
  globalError.hidden = false;
}

function clearError() {
  globalError.hidden = true;
  globalError.textContent = "";
}

function showLogin() {
  if (documentPoller) clearInterval(documentPoller);
  documentPoller = null;
  documents = [];
  selectedSourceButton = null;
  renderDocuments();
  chatContent.innerHTML = "";
  sourcePanel.innerHTML = `<div class="source-empty"><span class="section-label">Sources</span>
    <p>Source passages will appear here when you ask a question.</p></div>`;
  clearError();
  appShell.hidden = true;
  authScreen.hidden = false;
  passwordInput.focus();
}

function showApp() {
  authScreen.hidden = true;
  appShell.hidden = false;
  renderWelcome();
  refreshDocuments();
  if (!documentPoller) documentPoller = setInterval(refreshDocuments, 1200);
}

async function initialize() {
  try {
    const response = await fetch("/api/auth/status");
    const result = await response.json();
    if (response.ok && result.authenticated) showApp();
    else showLogin();
  } catch {
    showLogin();
    loginError.textContent = "Cannot connect to the local app server.";
    loginError.hidden = false;
  }
}

function updateQuestionControls() {
  const canAsk = documents.some((document) => document.status === "ready");
  questionInput.disabled = !canAsk || isAsking;
  askButton.disabled = !canAsk || isAsking;
  questionInput.placeholder = canAsk
    ? "Ask a question about your documents"
    : "Add a PDF first to start asking";
}

function renderDocuments() {
  documentCount.textContent = documents.length ? `(${documents.length})` : "";
  dropZone.hidden = documents.length > 0;
  addMoreButton.hidden = documents.length === 0;
  documentList.innerHTML = documents.map((document) => {
    const stateLabel = document.status === "ready" ? "Ready"
      : document.status === "indexing" ? "Indexing" : "Can't read";
    const metadata = document.status === "ready"
      ? `${document.page_count} pages · ${document.chunk_count} chunks`
      : document.status === "indexing"
        ? `Indexing: ${document.progress}%`
        : "PDF could not be indexed.";
    const progress = document.status === "indexing"
      ? `<div class="progress-track"><div class="progress-bar" style="width:${document.progress}%"></div></div>`
      : "";
    const error = document.error
      ? `<div class="document-error">${escapeHtml(document.error)}</div>` : "";
    const removeDisabled = document.status === "indexing" ? "disabled" : "";
    return `<article class="document-card">
      <div class="document-top">
        <span class="document-name">${escapeHtml(document.filename)}</span>
        <button class="remove-button" type="button" data-remove="${escapeHtml(document.id)}"
                aria-label="Remove ${escapeHtml(document.filename)}" title="Remove PDF"
                ${removeDisabled}>×</button>
      </div>
      <div class="document-meta">${metadata}</div>${progress}${error}
      <span class="status status-${escapeHtml(document.status)}">${stateLabel}</span>
    </article>`;
  }).join("");
  documentList.querySelectorAll("[data-remove]").forEach((button) => {
    button.addEventListener("click", () => removeDocument(button.dataset.remove));
  });
  updateQuestionControls();
}

function renderWelcome() {
  chatContent.innerHTML = `<section class="welcome">
    <h1>Ask questions about your own PDFs</h1>
    <p>Every answer shows where in your documents it came from.</p>
    <div class="steps">
      <div class="step-card"><strong>1. Add</strong>Drop one or more PDFs into the sidebar.</div>
      <div class="step-card"><strong>2. Wait for Ready</strong>Each file is read and indexed on your computer.</div>
      <div class="step-card"><strong>3. Ask</strong>Type a question and check the sources.</div>
    </div>
  </section>`;
}

async function refreshDocuments() {
  try {
    const response = await fetch("/api/documents");
    if (response.status === 401) {
      showLogin();
      return;
    }
    if (!response.ok) throw new Error(await response.text());
    documents = await response.json();
    renderDocuments();
  } catch (error) {
    showError(error.message || "Cannot connect to the local app.");
  }
}

async function uploadFiles(files) {
  const pdfs = [...files].filter((file) => file.name.toLowerCase().endsWith(".pdf"));
  if (!pdfs.length) {
    showError("Choose PDF files to add.");
    return;
  }
  clearError();
  const formData = new FormData();
  pdfs.forEach((file) => formData.append("files", file));
  try {
    const response = await fetch("/api/documents", { method: "POST", body: formData });
    const result = await response.json();
    if (response.status === 401) {
      showLogin();
      return;
    }
    if (!response.ok) throw new Error(result.detail || "PDF upload failed.");
    documents = [...documents, ...result];
    renderDocuments();
    chatContent.innerHTML = `<div class="conversation">
      <div class="notice-card">Your PDFs stay on this computer. They will be ready to search when indexing finishes.</div>
    </div>`;
    await refreshDocuments();
  } catch (error) {
    showError(error.message || "PDF upload failed.");
  } finally {
    fileInput.value = "";
  }
}

async function removeDocument(documentId) {
  clearError();
  try {
    const response = await fetch(`/api/documents/${encodeURIComponent(documentId)}`, {
      method: "DELETE",
    });
    const result = await response.json();
    if (response.status === 401) {
      showLogin();
      return;
    }
    if (!response.ok) throw new Error(result.detail || "Could not remove this PDF.");
    documents = documents.filter((document) => document.id !== documentId);
    renderDocuments();
    if (!documents.length) {
      renderWelcome();
      sourcePanel.innerHTML = `<div class="source-empty"><span class="section-label">Sources</span>
        <p>Source passages will appear here when you ask a question.</p></div>`;
    }
  } catch (error) {
    showError(error.message || "Could not remove this PDF.");
  }
}

function addMessage(question, answer, sources) {
  let conversation = chatContent.querySelector(".conversation");
  if (!conversation) {
    chatContent.innerHTML = '<div class="conversation"></div>';
    conversation = chatContent.querySelector(".conversation");
  }
  const turn = document.createElement("section");
  turn.className = "conversation-turn";
  turn.innerHTML = `<div class="user-message">${escapeHtml(question)}</div>
    <div class="answer-card">${formatAnswer(answer)}</div>`;
  const sourceCards = document.createElement("div");
  sourceCards.className = "message-sources";
  sources.forEach((source, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "source-chip";
    button.innerHTML = `<strong>${escapeHtml(source.filename)} · p. ${source.page}</strong>
      <span>${escapeHtml(source.text.slice(0, 100))}…</span>`;
    button.addEventListener("click", () => showSource(source, button));
    sourceCards.append(button);
    if (index === 0) setTimeout(() => button.click(), 0);
  });
  turn.append(sourceCards);
  conversation.append(turn);
  chatContent.scrollTop = chatContent.scrollHeight;
}

function showSource(source, button) {
  if (selectedSourceButton) selectedSourceButton.classList.remove("selected");
  selectedSourceButton = button;
  button.classList.add("selected");
  sourcePanel.innerHTML = `<div class="source-detail">
    <span class="section-label">Passage</span>
    <h2>${escapeHtml(source.filename)} · page ${source.page}</h2>
    <div class="passage">${escapeHtml(source.text)}</div>
    <a class="button button-outline open-page" target="_blank" rel="noopener"
       href="/api/documents/${encodeURIComponent(source.document_id)}\/file#page=${source.page}">
       Open full page
    </a>
  </div>`;
}

questionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || isAsking) return;
  clearError();
  isAsking = true;
  updateQuestionControls();
  const priorConversation = chatContent.querySelector(".conversation");
  if (!priorConversation) chatContent.innerHTML = '<div class="conversation"></div>';
  const conversation = chatContent.querySelector(".conversation");
  const loadingTurn = document.createElement("section");
  loadingTurn.className = "conversation-turn";
  loadingTurn.innerHTML = `<div class="user-message">${escapeHtml(question)}</div>
    <div class="answer-card loading">Finding relevant passages and asking Groq…</div>`;
  conversation.append(loadingTurn);
  questionInput.value = "";
  chatContent.scrollTop = chatContent.scrollHeight;
  try {
    const response = await fetch("/api/questions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const result = await response.json();
    if (response.status === 401) {
      loadingTurn.remove();
      showLogin();
      return;
    }
    if (!response.ok) throw new Error(result.detail || "Could not answer the question.");
    loadingTurn.remove();
    addMessage(question, result.answer, result.sources);
  } catch (error) {
    loadingTurn.querySelector(".answer-card").textContent =
      error.message || "Could not reach Groq. Check the API key and internet connection.";
    loadingTurn.querySelector(".answer-card").classList.remove("loading");
    showError(error.message || "Could not answer the question.");
  } finally {
    isAsking = false;
    updateQuestionControls();
  }
});

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginError.hidden = true;
  const button = loginForm.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: passwordInput.value }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Sign-in failed.");
    passwordInput.value = "";
    showApp();
  } catch (error) {
    loginError.textContent = error.message || "Sign-in failed.";
    loginError.hidden = false;
  } finally {
    button.disabled = false;
  }
});

logoutButton.addEventListener("click", async () => {
  try {
    const response = await fetch("/api/auth/logout", { method: "POST" });
    if (response.status === 401) {
      showLogin();
      return;
    }
    if (!response.ok) throw new Error("Could not sign out. Please try again.");
    showLogin();
  } catch (error) {
    showError(error.message || "Could not sign out.");
  }
});

addButton.addEventListener("click", () => fileInput.click());
addMoreButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => uploadFiles(fileInput.files));
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");
  uploadFiles(event.dataTransfer.files);
});

initialize();
