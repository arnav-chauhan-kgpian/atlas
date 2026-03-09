# Contributing to Atlas

Thanks for considering contributing to Atlas! Here's how to get started.

---

## 🛠️ Development Setup

```bash
# Clone the repo
git clone https://github.com/arnav-chauhan-kgpian/atlas.git
cd atlas

# Create a virtual environment
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS / Linux

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## 🧪 Running Tests

```bash
# Full test suite
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ -v --cov=atlas
```

All contributions should pass the existing test suite. If you're adding new functionality, please add corresponding tests.

## 📝 Code Style

- **Type hints** — All public functions and methods should use type annotations.
- **Docstrings** — Use Google-style docstrings for classes and functions.
- **Imports** — Use absolute imports (`from atlas.model.attention import ...`).
- **Naming** — Follow PEP 8: `snake_case` for functions/variables, `PascalCase` for classes.

## 🔀 Pull Requests

1. **Fork** the repository and create a feature branch from `main`.
2. **Make your changes** with clear, descriptive commits.
3. **Add tests** for any new functionality.
4. **Run the test suite** to ensure nothing is broken.
5. **Open a PR** with a clear description of what you changed and why.

## 🐛 Bug Reports

When filing a bug report, please include:
- Python and PyTorch versions
- Steps to reproduce
- Expected vs. actual behavior
- Full error traceback (if applicable)

## 💡 Feature Requests

Open an issue describing:
- The problem your feature solves
- Your proposed approach (if any)
- Any alternatives you've considered

---

Thank you for helping make Atlas better! 🚀
