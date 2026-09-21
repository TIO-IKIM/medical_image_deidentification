import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"


def _project_dependencies():
    text = PYPROJECT.read_text(encoding="utf-8")
    block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    return {
        re.match(r'\s*"([^"<>=!~;]+)', line).group(1)
        for line in block.splitlines()
        if re.match(r'\s*"([^"<>=!~;]+)', line)
    }


def _requirements_dependencies():
    return {
        re.split(r"[<>=!~;]", line, maxsplit=1)[0].strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _canonicalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


class TestPackagingConfiguration(unittest.TestCase):
    def test_package_sources_parse_as_python_310(self):
        for source_file in (ROOT / "mede").rglob("*.py"):
            ast.parse(
                source_file.read_text(encoding="utf-8"),
                filename=str(source_file),
                feature_version=(3, 10),
            )

    def test_supported_python_range_and_classifiers(self):
        text = PYPROJECT.read_text(encoding="utf-8")

        self.assertIn('requires-python = ">=3.10,<3.13"', text)
        for version in ("3.10", "3.11", "3.12"):
            self.assertIn(
                f'"Programming Language :: Python :: {version}"',
                text,
            )

    def test_requirements_and_project_have_the_same_direct_dependencies(self):
        project_names = {_canonicalize(name) for name in _project_dependencies()}
        requirements_names = {
            _canonicalize(name) for name in _requirements_dependencies()
        }

        self.assertEqual(project_names, requirements_names)

    def test_coupled_and_problematic_dependencies_are_constrained(self):
        dependencies = _project_dependencies()

        self.assertIn("scikit-image", dependencies)
        self.assertIn("torchvision", dependencies)
        text = PYPROJECT.read_text(encoding="utf-8")
        self.assertIn('"scikit-image>=0.24,<0.25"', text)
        self.assertIn('"torch==2.2.2"', text)
        self.assertIn('"torchvision==0.17.2"', text)
        self.assertIn('"pandas>=2,<3"', text)
        self.assertIn('"deid==0.4.12"', text)
        self.assertIn('"pydicom==3.0.1"', text)


if __name__ == "__main__":
    unittest.main()
