"""Keep BLAS single-threaded for this package's small dense solves.

Every rating model solves (or iterates on) a system with one row per team, a few hundred at
most, thousands of times in a walk-forward backtest or a grid search. Multi-threaded OpenBLAS
is the wrong tool for that: thread start-up and spin-waiting dominate a 135 x 135 solve, and
when two such processes share a machine (two backtests, a backtest next to the dashboard,
the test-suite) they starve each other -- measured at 1000 x slower per solve. Pinning the
BLAS thread pool to one thread makes every fit take milliseconds regardless of load.

The variables are only read when the BLAS library is loaded, i.e. when ``numpy`` is first
imported, so this module must run before that: it is the first import of :mod:`cfbratings`.
Callers who really want threaded BLAS can set the variables themselves beforehand.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
