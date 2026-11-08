"""Promo finder: reads microdrama screenplays and proposes ranked 20-60s promo clips.

The pipeline is model-agnostic: every LLM sits behind `promo_finder.llm.LLM`,
selected at runtime with a spec string such as "groq:openai/gpt-oss-120b".
"""

__version__ = "0.1.0"
