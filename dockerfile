# Dockerfile — install minimal system libs for OpenCV
FROM python:3.11-slim

WORKDIR /app

# Copy only what we need
COPY requirements.txt /app/
COPY app.py /app/

# Install apt deps required by OpenCV + cleanup to keep image small
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6 \
  libgl1 \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip then install python deps
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Ensure a .streamlit config exists in container
RUN mkdir -p /app/.streamlit \
  && printf "[server]\nheadless = true\nport = 8501\nenableCORS = false\n" > /app/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
