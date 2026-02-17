# Johnson's Relative Weights - Web Application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY app.py .
COPY weights_handler.py .
COPY spss_handlers.py .
COPY johnson_weights.py .
COPY file_handlers/ ./file_handlers/

# Copy templates and static files
COPY templates/ ./templates/
COPY static/ ./static/

# Copy documentation
COPY README.md .
COPY ["Multiple Imputations Readme.md", "."]

# Copy sample data if exists (optional)
COPY sample_data.sav* ./

# Create directories for uploads and results
RUN mkdir -p temp results

# Expose port (Koyeb will set PORT env variable)
EXPOSE 8000

# Run with gunicorn for production
# Using 1 worker because jobs are stored in memory (not shared between workers)
# With multiple workers, a job created in worker 1 won't be found by worker 2
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
