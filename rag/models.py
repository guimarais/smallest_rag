from typing import Generator
import ollama


def check_ollama_running() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False


def model_exists(name: str) -> bool:
    try:
        ollama.show(name)
        return True
    except ollama.ResponseError:
        return False
    except Exception:
        return False


def pull_model(name: str) -> Generator:
    """Yields ProgressResponse objects from ollama.pull stream."""
    for progress in ollama.pull(name, stream=True):
        yield progress
