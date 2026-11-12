FROM python@sha256:3fe1a0b6c2d6b2ebb572906cdabeca75aeb558c0c17db5ffefbcf8094871c32b
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY --chown=10001:10001 src/__init__.py src/model.py src/smoke.py ./src/
COPY --chown=10001:10001 models/mnist_linear.npz ./models/mnist_linear.npz
COPY --chown=10001:10001 metrics/model.sha256 ./metrics/model.sha256
USER 10001:10001
HEALTHCHECK --interval=30s --timeout=10s --retries=2 CMD ["python", "-m", "src.smoke"]
CMD ["python", "-m", "src.smoke"]
