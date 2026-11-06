"""Start the command-line interface when the package is run as a module.

This file is the entry point for commands such as
``python -m public_sector_opportunity_research init-db``. Python first imports
the package, then executes this module. The module delegates all argument
definition and command handling to :func:`public_sector_opportunity_research.cli.main`.

Nothing comes before this file except Python's package import machinery. After
``main`` receives the command, ``cli.py`` creates configuration, storage,
chunking, and ingestion objects as needed. Keeping this wrapper tiny means the
same ``cli.main`` function can also be called by the installed console script
or by tests.
"""

from .cli import main


if __name__ == "__main__":
    # Convert main's integer return value into the process exit status.
    raise SystemExit(main())

