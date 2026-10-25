"""
`python -m ocralign.backends.paddle up|down|status` runs service/stack.sh.

The stack script ships inside the package, so this finds the installed copy
without anyone capturing an import's stdout to learn its path.
"""

import os
import sys

from ocralign.backends.paddle import STACK_SCRIPT

os.execvp("bash", ["bash", STACK_SCRIPT, *sys.argv[1:]])
