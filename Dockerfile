# Use an official Python runtime as the base image
FROM python:3.9-slim

# Install dependencies
RUN pip install google-play-scraper pandas

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt first to cache dependencies
COPY requirements.txt /app/

# Install the required dependencies
RUN pip install -r requirements.txt

# Copy the rest of the content into the container
COPY . /app/

# Run the script that ties everything together (run_all.py)
CMD ["python", "run_all.py"]
