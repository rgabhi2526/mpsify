"""Wrapper: `python -m mpsify [--profile] <script> [args...]`

Sets PYTORCH_ENABLE_MPS_FALLBACK before torch loads, installs the shim,
then runs the target script as __main__ with its own argv.
"""
import os
import runpy
import sys


def main() -> None:
    argv = sys.argv[1:]

    if argv and argv[0] == "doctor":
        from ._doctor import doctor
        if len(argv) < 2:
            print("usage: python -m mpsify doctor <script.py>", file=sys.stderr)
            sys.exit(2)
        sys.exit(doctor(argv[1]))

    profile = False
    if argv and argv[0] == "--profile":
        profile = True
        argv = argv[1:]

    if not argv:
        print("usage: python -m mpsify [--profile] <script.py> [args...]\n"
              "       python -m mpsify doctor <script.py>", file=sys.stderr)
        sys.exit(2)

    # Must precede any torch import.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import mpsify  # triggers patch()

    script = argv[0]
    sys.argv = argv  # target sees: [script, its, args]

    if profile:
        from . import _report
        with _report.profiler():
            runpy.run_path(script, run_name="__main__")
    else:
        runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
