FROM python:slim

RUN apt-get update && apt-get install -y default-mysql-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose the port that Streamlit runs on
EXPOSE 8501
# Command to run the web application
CMD ["streamlit", "run", "app1.py"]
