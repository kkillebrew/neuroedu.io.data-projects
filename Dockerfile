# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install all the heavy data science dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose port 8080 (DigitalOcean App Platform standard)
EXPOSE 8080

# Start Streamlit, pointing it to your main Data Projects Hub file
CMD ["streamlit", "run", "data_projects_app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]