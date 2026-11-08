FROM python:3.11-slim

WORKDIR /app

COPY requirements-lite.txt .
RUN pip install --no-cache-dir -r requirements-lite.txt

COPY app.py router.py ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
