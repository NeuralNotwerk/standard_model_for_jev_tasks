"""Jev-compatible decision-model shim for standard LLMs (vLLM / llama.cpp)."""

from .server import (Backend, LlamaCppBackend, ShimError, answer_question,  # noqa: F401
                     build_grammar, build_label_grammar, build_lines_grammar,
                     build_pick_grammar, labels_and_options, main, parse_lines, pick_answer)

from .remote import (BedrockBackend, OpenAICompatBackend, answer_pick,  # noqa: F401,E402
                     detect_backend_type)

__version__ = "0.2.0"
