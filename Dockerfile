FROM python:3.11-slim
WORKDIR /app

# Copy requirement list and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port 8080 for DigitalOcean
EXPOSE 8080

# Force Streamlit to port 8080 for DigitalOcean Health Checks
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]