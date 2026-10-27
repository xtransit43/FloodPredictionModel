FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for scientific packages (removed software-properties-common)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files, models, and source code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Configure Streamlit to run headlessly in containers
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

CMD ["streamlit", "run", "app.py"]
