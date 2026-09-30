FROM python:3.10-slim

# Install system dependencies required by OpenCV and Git for LightGlue
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the project files with the correct permissions
COPY --chown=user:user . /app/

# Install AeroMetric and its dependencies (including LightGlue via git)
RUN pip install --no-cache-dir .[dev,matching]

# Create the runs directory where uploaded videos will go
RUN mkdir -p /app/runs

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Start the web app on 0.0.0.0 so the Docker container can receive external requests
CMD ["python", "-m", "aerometric", "app", "--runs", "/app/runs", "--host", "0.0.0.0", "--port", "7860"]
