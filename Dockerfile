# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt /app/

RUN pip install pip \
    && pip install -r requirements.txt \
    && rm -rf /root/.cache

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
COPY requirements.txt /app/


# Copy the rest of the application
COPY . /app/

# Expose the port the app runs on
EXPOSE 8000

ENV PYTHONPATH=/app

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
