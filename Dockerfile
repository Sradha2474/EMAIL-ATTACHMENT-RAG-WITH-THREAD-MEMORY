FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install pymupdf==1.24.3 --only-binary=:all: && \
    pip install -r requirements.txt

COPY . .

RUN python ingest.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]