FROM python:3.9-slim
WORKDIR /app
# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy all application files
COPY train_model.py predict_cli.py app.py ./
# Create models directory
RUN mkdir -p models
# Set environment variable to ensure Streamlit doesn't try to create a config file
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
# Train the model when building the image
RUN python train_model.py
# Expose the port that Streamlit runs on
EXPOSE 8501
# Command to run the web application
CMD ["streamlit", "run", "app.py"]