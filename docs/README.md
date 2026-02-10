# Documentation for python_magnetcooling

This directory contains the Sphinx documentation for the python_magnetcooling package.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

This will install:
- sphinx
- sphinx-rtd-theme
- sphinx-autodoc-typehints

### Build HTML Documentation

On Linux/macOS:

```bash
cd docs
make html
```

On Windows:

```bash
cd docs
make.bat html
```

The generated HTML documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Other Build Targets

- `make clean` - Remove build artifacts
- `make latexpdf` - Build PDF documentation (requires LaTeX)
- `make linkcheck` - Check all external links
- `make doctest` - Run doctests in the documentation

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── installation.rst     # Installation instructions
├── quickstart.rst       # Quick start guide
├── api/                 # API reference
│   ├── index.rst
│   └── *.rst            # Module documentation
├── examples/            # Usage examples
│   ├── index.rst
│   └── *.rst
├── theory/              # Theoretical background
│   ├── index.rst
│   └── *.rst
├── _static/             # Static files (images, CSS)
└── _templates/          # Custom templates
```

## Contributing to Documentation

When adding new modules or features:

1. Add docstrings to your Python code
2. Create/update relevant .rst files in the appropriate directory
3. Update the table of contents in index files
4. Build and test the documentation locally
5. Check for warnings during the build

## Auto-documentation

The API documentation is automatically generated from docstrings using Sphinx's autodoc extension. Make sure your code includes comprehensive docstrings following NumPy or Google style.
