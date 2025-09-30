"""Wrapper to run pytest programmatically and write output to stdout/stderr.
This avoids conda run subprocess buffering issues when launching pytest via shell wrappers.
"""
import sys

import pytest

if __name__ == '__main__':
    # run pytest with arguments passed from command line
    # we keep pytest's own exit code and propagate it
    sys.exit(pytest.main(sys.argv[1:]))
