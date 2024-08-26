# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Set the environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8501

# Run Streamlit
CMD ["streamlit", "run", "main4.py"]
