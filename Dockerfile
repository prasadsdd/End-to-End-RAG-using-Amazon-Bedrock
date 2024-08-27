# Use the official Python image from the Docker Hub
FROM python:3.11

# Expose the port Streamlit uses
EXPOSE 8083

# Set the working directory
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt ./

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . ./

ENTRYPOINT [ "streamlit", "run", "main.py", "--server.port=8083", "--server.address=0.0.0.0" ]