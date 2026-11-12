FROM python@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f
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
