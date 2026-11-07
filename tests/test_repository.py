"""Repository scaffold checks only; these do not certify animation quality."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_required_entries_exist(self):
        for relative in (
            "README.md", "AGENTS.md", "docs/BOUNDARIES.md", "docs/PLAN.md",
            "runtime/README.md", "preview/README.md", "tools/README.md",
            "assets/README.md", ".gitignore", ".editorconfig", ".gitattributes",
        ):
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_scaffold_text_is_utf8_without_bom(self):
        files = list(ROOT.rglob("*.md")) + list(ROOT.rglob("*.py"))
        for path in files:
            if ".git" in path.relative_to(ROOT).parts:
                continue
            with self.subTest(path=str(path.relative_to(ROOT))):
                data = path.read_bytes()
                self.assertFalse(data.startswith(b"\xef\xbb\xbf"))
                self.assertTrue(data.decode("utf-8").endswith("\n"))


if __name__ == "__main__":
    unittest.main()
