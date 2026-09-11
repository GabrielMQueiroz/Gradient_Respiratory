# Contributing to sttensor

Thank you for your interest in contributing to `sttensor`! We welcome contributions, bug reports, feature requests, and documentation improvements from the community.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all contributors with respect and kindness.

---

## Getting Started

### 1. Fork and Clone the Repository

```bash
git clone https://github.com/GabrielMQueiroz/Gradient_Respiratory.git
cd Gradient_Respiratory
```

### 2. Set Up a Virtual Environment

We recommend using Python 3.8 or higher.

```bash
# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install in Editable Mode with Development Dependencies

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

If you plan to run video demos and real-time dashboards:
```bash
pip install -e ".[all]"
```

---

## Running the Test Suite

All contributions must pass existing tests and include new tests where applicable.

```bash
# Run the complete test suite
pytest

# Run with verbose output and coverage
pytest -v
```

---

## Project Architecture

```text
RespFreq/
├── sttensor/             # Core library package
│   ├── gradients.py      # Spatial & temporal gradient differences
│   ├── tensor.py         # Structure tensor computation
│   ├── eigen.py          # Eigen-decomposition & coherence metrics
│   ├── filters.py        # Scale-adaptive & directional filters
│   ├── multiscale.py     # Multi-scale pyramid & fusion
│   ├── benchmarks.py     # Edge spread & benchmark utilities
│   └── utils.py          # Video I/O & visualization
├── tests/                # Unit & integration test suite (pytest)
├── apps/                 # Real-time processing pipelines & web apps
├── experiments/          # Reproducible validation & benchmark scripts
├── notebooks/            # Interactive Jupyter notebooks
├── paper/                # Research paper LaTeX source & bibliography
└── docs/                 # Detailed mathematical specifications
```

---

## Coding Guidelines

- **PEP 8**: Follow standard Python style guidelines.
- **Type Annotations**: Use `typing` hints (`np.ndarray`, `Tuple`, `Optional`, etc.) for public functions.
- **Docstrings**: Provide clear docstrings explaining inputs, outputs, shapes, and formulas.
- **Pure Numerical Functions**: Mathematical operations in `sttensor/` should avoid unnecessary side effects and work on arbitrary 2D/3D NumPy arrays.

---

## Submitting Pull Requests

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Commit your changes**:
   Write clear, concise commit messages following the Conventional Commits format (e.g., `feat: add robust tensor trace thresholding`, `fix: handle singular matrix in eigen decomposition`).
3. **Run tests**:
   Ensure all tests pass before submitting (`pytest`).
4. **Push and open a PR**:
   Push your branch to your fork and submit a Pull Request describing your changes, motivation, and test coverage.

---

## Reporting Issues

If you encounter a bug or have a suggestion, please open an issue on GitHub:
- [Bug Report](https://github.com/GabrielMQueiroz/Gradient_Respiratory/issues/new?template=bug_report.md)
- [Feature Request](https://github.com/GabrielMQueiroz/Gradient_Respiratory/issues/new?template=feature_request.md)

Please provide a minimal reproducible example for bug reports whenever possible.
