.PHONY: install lint format test test-fast demo backtest app clean

install:        ## editable install with dashboard + dev tools
	pip install -e ".[dev]"

lint:           ## static checks
	ruff check .
	ruff format --check .

format:         ## auto-format
	ruff format .
	ruff check --fix .

test:           ## full test-suite
	pytest

test-fast:      ## skip the slow dashboard smoke test
	pytest -m "not slow"

demo:           ## rankings on a synthetic season (no API key needed)
	cfbratings --demo rank --method hybrid --top 25

backtest:       ## walk-forward comparison of every method on a synthetic season
	cfbratings --demo backtest --start-week 4

app:            ## launch the dashboard (BLAS pinned to one thread: see cfbratings/_blas.py)
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 streamlit run apps/streamlit_app.py

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
