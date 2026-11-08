import sys

try:
    from .cli import main  # importing reads and checks the settings in .env
except ValueError as e:  # an invalid setting: show the message, not a traceback
    print(f"error: {e}", file=sys.stderr)
    raise SystemExit(2)

raise SystemExit(main())
