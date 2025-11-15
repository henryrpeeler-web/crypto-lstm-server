# Use a Python 3.12 slim image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files from repo into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 10000 (matches FastAPI port)
EXPOSE 10000

# Run FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]
