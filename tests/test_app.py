import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app


class Matrix(list):
    def tolist(self):
        return list(self)


class FakeEmbedder:
    def encode(self, texts):
        return Matrix([[float(number)] for number, _ in enumerate(texts)])


class FakeCollection:
    def add(self, **values):
        self.values = values

    def count(self):
        return len(self.values["documents"])

    def query(self, n_results, **_kwargs):
        return {
            "documents": [self.values["documents"][:n_results]],
            "metadatas": [self.values["metadatas"][:n_results]],
            "distances": [[0.1] * n_results],
        }


class RagPipelineTests(unittest.TestCase):
    def tearDown(self):
        app._embedder = app._collection = None

    def test_clean_text_normalizes_ligatures_and_preserves_paragraphs(self):
        text = "The o\uFB03ce con-\ntains  2\nlines.\n\nNew para."
        self.assertEqual(
            app.clean_text(text),
            "The office contains 2 lines.\n\nNew para.",
        )

    def test_chunk_text_uses_character_overlap(self):
        self.assertEqual(
            app.chunk_text("abcdefghij", chunk_size=5, overlap=2),
            ["abcde", "defgh", "ghij"],
        )

    def test_chunk_text_rejects_overlap_equal_to_chunk_size(self):
        with self.assertRaises(ValueError):
            app.chunk_text("some text", chunk_size=5, overlap=5)

    def test_cli_settings_are_configurable(self):
        args = app.parse_args([
            "pdfs", "--recursive", "--chunk-size", "700", "--overlap", "120",
            "--top-k", "5",
        ])
        self.assertTrue(args.recursive)
        self.assertEqual((args.chunk_size, args.overlap, args.top_k), (700, 120, 5))

    def test_index_retrieval_and_prompt_keep_source_citations(self):
        collection = FakeCollection()
        client = SimpleNamespace(
            list_collections=lambda: [SimpleNamespace(name=app.COLLECTION_NAME)],
            delete_collection=lambda name: self.assertEqual(name, app.COLLECTION_NAME),
            create_collection=lambda name: collection,
        )
        with patch("app.chromadb.PersistentClient", return_value=client), \
                patch("app.SentenceTransformer", return_value=FakeEmbedder()):
            app.build_index(
                ["first passage", "second passage"],
                [("chapter-one.pdf", 4), ("nested/chapter-two.pdf", 9)],
            )

        self.assertEqual(collection.values["ids"], ["chunk_0", "chunk_1"])
        self.assertEqual(collection.values["metadatas"][1], {
            "chunk_number": 1,
            "source_file": "nested/chapter-two.pdf",
            "page_number": 9,
        })
        results = app.retrieve("second passage", k=1)
        self.assertEqual(results[0][1:4], (0, "chapter-one.pdf", 4))
        prompt = app.build_prompt("question", results)
        self.assertIn("chapter-one.pdf, page 4, chunk 0", prompt)
        self.assertIn("Answer using ONLY the context below.", prompt)
        self.assertIn("Use simple words, short sentences, and short paragraphs.", prompt)
        self.assertIn("'I could not find this in the document.'", prompt)


if __name__ == "__main__":
    unittest.main()
