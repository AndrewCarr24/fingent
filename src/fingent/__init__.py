"""FinGent — LangGraph QA agent for knowledge bases.

Public API:
    >>> from fingent import Agent, build_kb
    >>> build_kb(parsed_dir="./my_docs", kb_id="my_kb", store_dir="./fingent_store")
    >>> agent = Agent(kb_id="my_kb", store_dir="./fingent_store")
    >>> async for chunk in agent.ask_stream("..."):
    ...     print(chunk, end="")
"""

from fingent.agent import Agent
from fingent.build_kb import build_kb

__version__ = "0.1.0"
__all__ = ["Agent", "build_kb", "__version__"]
