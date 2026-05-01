FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY project/   ./project/
COPY TinyLlama/ ./TinyLlama/
COPY LoRA/      ./LoRA/

# model.py resolves paths via Path(__file__).parent.parent → /app
# Run from /app/project so `import model as m` resolves correctly
WORKDIR /app/project

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
