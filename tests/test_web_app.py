import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_app


class FakeCollection:
    def count(self):
        return 1

    def query(self, **_kwargs):
        return {
            "documents": [["A useful passage."]],
            "metadatas": [[{
                "chunk_number": 2,
                "document_id": "doc-1",
                "source_file": "notes.pdf",
                "page_number": 7,
            }]],
            "distances": [[0.15]],
        }

    def delete(self, **_kwargs):
        return None


class FakeMatrix(list):
    def tolist(self):
        return list(self)


class FakeEmbedder:
    def encode(self, texts):
        return FakeMatrix([[0.1] for _ in texts])


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web_app.app)
        self.old_documents = web_app._documents
        self.old_collection = web_app._collection
        self.old_embedder = web_app._embedder
        web_app._documents = {
            "doc-1": {
                "id": "doc-1", "filename": "notes.pdf", "stored_name": "doc-1.pdf",
                "status": "ready", "page_count": 8, "chunk_count": 3,
                "progress": 100, "error": "",
            }
        }
        web_app._collection = FakeCollection()
        web_app._embedder = FakeEmbedder()

    def tearDown(self):
        self.client.close()
        web_app._documents = self.old_documents
        web_app._collection = self.old_collection
        web_app._embedder = self.old_embedder

    def test_serves_interface_and_static_assets(self):
        self.assertIn("Chat with your PDFs", self.client.get("/").text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)

    def test_startup_loads_project_env_file(self):
        with patch.object(web_app, "load_dotenv") as load_env, \
                patch.object(web_app, "_load_documents"):
            with TestClient(web_app.app):
                load_env.assert_called_once_with(dotenv_path=Path(web_app.__file__).resolve().parent / ".env")

    def test_question_response_includes_document_id_and_page_citation(self):
        with patch.object(web_app.rag, "ask_llm", return_value="The answer."):
            response = self.client.post(
                "/api/questions", json={"question": "What is useful?"}
            )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["answer"], "The answer.")
        self.assertEqual(result["sources"][0]["document_id"], "doc-1")
        self.assertEqual(result["sources"][0]["filename"], "notes.pdf")
        self.assertEqual(result["sources"][0]["page"], 7)
        self.assertIn("notes.pdf, page 7", web_app.rag.build_prompt(
            "What is useful?", [("A useful passage.", 2, "notes.pdf", 7, 0.15)]
        ))

    def test_question_requires_a_nonempty_question(self):
        response = self.client.post("/api/questions", json={"question": "  "})
        self.assertEqual(response.status_code, 400)

    def test_upload_saves_local_file_and_persists_document_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_dir = web_app.Path(directory)
            manifest_path = upload_dir / "documents.json"
            with patch.object(web_app, "UPLOAD_DIR", upload_dir), \
                    patch.object(web_app, "MANIFEST_PATH", manifest_path), \
                    patch.object(web_app, "_index_document"):
                web_app._documents = {}
                response = self.client.post(
                    "/api/documents",
                    files={"files": ("study-notes.pdf", b"%PDF-test-content", "application/pdf")},
                )
                self.assertEqual(response.status_code, 200, response.text)
                document = response.json()[0]
                self.assertEqual(document["filename"], "study-notes.pdf")
                self.assertEqual(document["status"], "indexing")
                self.assertTrue(manifest_path.is_file())
                stored_path = upload_dir / f"{document['id']}.pdf"
                self.assertEqual(stored_path.read_bytes(), b"%PDF-test-content")
                file_response = self.client.get(f"/api/documents/{document['id']}/file")
                self.assertEqual(file_response.content, b"%PDF-test-content")


if __name__ == "__main__":
    unittest.main()
