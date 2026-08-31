"""Shared play-by-play parsing used by both analysis pipelines.

An explicit __init__.py rather than a PEP-420 namespace package: without it mypy
refuses to run at all, reporting `Source file found twice under different module
names: "pbp" and "common.pbp"`, because both pipelines put `src/` on sys.path and
the module is then reachable under two names.
"""
