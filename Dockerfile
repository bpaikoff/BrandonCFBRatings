# CFB Ratings dashboard — the image Railway builds for https://cfb.tzadek.ai
#
# Build and run locally (then open http://localhost:8501):
#   docker build -t cfbratings .
#   docker run --rm -p 8501:8501 -e CFBD_API_KEY=... cfbratings
#
# Railway injects $PORT; the CMD binds to it (8501 when unset). The committed archive of
# finished seasons rides along, so the only API calls the container ever makes are for the
# season in progress. data/cache is a working cache: mount a volume there to keep it across
# deploys, or leave it ephemeral (a cold container re-fetches the current season once).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # small dense solves: single-threaded BLAS is faster and never starves (cfbratings/_blas.py)
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    # production server: bind every interface, no file watcher, no "Deploy" prompts
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # anyone can open the site: no refresh button, archived seasons + the current one only, so
    # visitors cannot spend the CFBD key's 1 000 calls a month (override with 0 on Railway)
    CFB_PUBLIC_DASHBOARD=1

WORKDIR /app

# Install the package and its dashboard extras first so the dependency layer is cached
# until pyproject.toml or the package itself changes.
COPY pyproject.toml README.md ./
COPY cfbratings/ ./cfbratings/
RUN pip install ".[app]"

# The dashboard, what it renders (docs/METHODOLOGY.md), its theme and the season archive.
COPY .streamlit/ ./.streamlit/
COPY apps/ ./apps/
COPY docs/METHODOLOGY.md ./docs/
COPY data/archive/ ./data/archive/
RUN mkdir -p data/cache

EXPOSE 8501

# Streamlit's health endpoint; Railway polls the same path (.railway/railway.ts).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8501') + '/_stcore/health')" || exit 1

# Shell form so $PORT expands.
CMD streamlit run apps/streamlit_app.py --server.port "${PORT:-8501}"
