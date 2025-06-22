### Example API & Streamlit Workflow

#### Useful Commands

[uv is great lol - give it a try](https://docs.astral.sh/uv/getting-started/installation/) below is how you can do it on MacOS

```bash
brew install uv
```

Using uv if you have an existing requiremnets.txt and no pyproject files:

```bash
uv add -r requirements.txt
```
- This allowed us to create a uv environment using an existing `requirements.txt`


Otherwise, if you do, simply run

```bash
uv sync
```

- This will create a uv environment using the existing `pyproject.toml` and `uv.lock`

```bash
uv run mypy --strict --check-untyped-defs api 
```

- This will use `mypy` to type check your `api` folder.

