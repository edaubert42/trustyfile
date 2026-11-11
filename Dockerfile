# syntax=docker/dockerfile:1
#
# TrustyFile: one image for production (Railway) and for CI.
#
# Production and the test suite used to be prepared separately: Railpack built
# production from service variables, while CI installed its own packages on an
# Ubuntu runner. This file is now the only recipe for both: CI runs pytest in
# the `test` stage and boots the `runtime` stage. Railway builds the `runtime`
# stage, because it is the last one and Railway builds without --target
# (railway.json selects this Dockerfile).
#
# Stages:
#   caddy    official Caddy image, only used to copy its binary from
#   app      Python, system libraries, runtime dependencies, the site
#   test     app + test dependencies + tests/ (CI: docker build --target test)
#   runtime  app as it is, built by Railway (keep it LAST)

# Python image of production. CI also builds the image on the next Python with
# --build-arg PYTHON_IMAGE=python:3.14.x-slim-trixie. Not named PYTHON_VERSION:
# Railway passes any service variable whose name matches an ARG to the build,
# and that name is common enough to exist already.
ARG PYTHON_IMAGE=python:3.13.18-slim-trixie

# Caddy, one exact release. Its binary is static (no libc dependency), so it
# runs as is on Debian. To upgrade: change this tag.
FROM caddy:2.11.9-alpine AS caddy


FROM ${PYTHON_IMAGE} AS app

# PYTHONUNBUFFERED: logs reach Railway as they are written.
# PORT: Railway sets its own; the default is for `docker run` and CI.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

# System libraries the Python wrappers bind to:
#   tesseract-ocr(-fra)  pytesseract calls the `tesseract` binary (OCR
#                        fallback for flattened PDFs, French documents)
#   libdmtx0t64          pylibdmtx loads libdmtx.so.0 (2D-Doc DataMatrix).
#                        Debian 13 name; Debian 12 called it libdmtx0b.
# Not installed, on purpose: libzbar0 (QR codes are read with
# cv2.QRCodeDetector, pyzbar is not a dependency), libgl1 (we ship
# opencv-python-headless, which does not link libGL), curl (nothing calls it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-fra \
        libdmtx0t64 \
    && rm -rf /var/lib/apt/lists/*

# At a fixed path that start.sh calls directly: no other caddy can shadow it.
COPY --from=caddy /usr/bin/caddy /usr/local/bin/caddy

WORKDIR /app

# Dependencies before the code: a code-only change reuses this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the site needs at run time. tests/, scripts/, docs/ stay out of
# the production image; samples/ (real documents) never even reaches the
# build context, see .dockerignore.
COPY Caddyfile start.sh ./
COPY src/ src/
COPY public/ public/

# Compile the bytecode now rather than on the first request: the service
# sleeps when idle, so every wake-up is a cold start.
RUN python -m compileall -q src

EXPOSE 8080
CMD ["bash", "start.sh"]


FROM app AS test

COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# The whole build context (minus .dockerignore): tests import scripts/ and
# read files at the repository root.
COPY . .

CMD ["python", "-m", "pytest", "tests/", "-q"]


# What Railway builds and runs. Must stay the last stage.
FROM app AS runtime
