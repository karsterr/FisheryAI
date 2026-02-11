# Contributing to FisheryAI

Thank you for your interest in contributing to FisheryAI! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/FisheryAI.git
   cd FisheryAI
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up the environment** (Docker recommended):
   ```bash
   docker build -t fisheryai .
   ```
   Or locally with Python 3.10/3.11:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## What You Can Contribute

### Bug Fixes
- Improve error handling and edge cases
- Fix input validation issues
- Address platform-specific bugs (Windows, macOS, Linux)

### Features
- New CLI options or modes
- Additional output formats (JSON, CSV)
- Localization / multi-language support
- Dashboard or web interface (Streamlit, Flask, etc.)
- Integration with external systems (n8n, email alerts)
- Accessibility improvements

### Documentation
- Improve README or inline comments
- Add usage examples
- Translate documentation

### Testing
- Add unit tests for the `FisheryAI` class
- Add integration tests for CLI modes
- Add test data scenarios

## Important Rules

### Do NOT Modify the Trained Model

The following files are the trained model artifacts and **must not be modified**:

- `fishery_model.tflite` -- the quantized TFLite model
- `model_meta.json` -- the tokenizer vocabulary and topic mapping

If you want to improve the model, work in `model_development.ipynb` and train a new version. Submit the new `.tflite` and `.json` files as part of your PR with a clear explanation of the changes and accuracy metrics.

### Code Style

- Keep it simple. This is a focused, lightweight project.
- Use clear variable names and add comments for non-obvious logic.
- Preserve the existing coding style (PEP 8, 4-space indentation).
- Error messages should be informative and actionable.
- The `FisheryAI` class should remain usable as a library (no `exit()` calls, no side effects on import).

### Commit Messages

- Use concise, descriptive commit messages.
- Start with a verb: `fix`, `add`, `update`, `remove`, `refactor`.
- Example: `fix: handle NaN values in data input validation`

## Submitting Changes

1. **Test your changes** before submitting:
   ```bash
   # Quick batch test
   python pc_inference.py --batch 4 70,65,55,48,40
   python pc_inference.py --batch 1 25000,28000,31000,35000,42000
   python pc_inference.py --batch 2 1000,1500,2500,4000,6500
   python pc_inference.py --batch 3 12.5,13.0,14.2,15.8,18.5

   # Or with Docker
   docker build -t fisheryai . && docker run --rm fisheryai --batch 4 70,65,55,48,40
   ```

2. **Commit** your changes:
   ```bash
   git add .
   git commit -m "add: brief description of your change"
   ```

3. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** on GitHub against the `main` branch. Include:
   - A clear description of what changed and why
   - Test results showing the change works
   - Any relevant screenshots for UI changes

## Reporting Issues

When reporting a bug, please include:

- Python version (`python --version`)
- Operating system
- Full error output / traceback
- Steps to reproduce
- Input data that triggered the issue

## Questions?

Open an issue on GitHub with the `question` label.
