"""Identify the package and expose its version number.

This is the smallest module in the RAG foundation. Python runs it when code
first imports :mod:`public_sector_opportunity_research`, before the requested
submodule such as ``cli`` or ``storage`` is loaded. It does not ingest,
retrieve, or generate anything itself; it simply marks this directory as the
application package and gives packaging or diagnostic code one stable place to
read the current version.

In the command-line workflow, Python imports this package before
``__main__.py``. Control then moves to ``cli.main`` and the rest of the RAG
pipeline. No application module depends on ``__version__`` at present.
"""

# Keep the version as a plain string so packaging and debugging tools can read it.
__version__ = "0.1.0"

