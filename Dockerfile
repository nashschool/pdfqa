# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Update system and install gcc and other necessary dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    build-essential \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Expose the port the app runs on
EXPOSE 8000

ENV PYTHONPATH="${PYTHONPATH}:/app"

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
