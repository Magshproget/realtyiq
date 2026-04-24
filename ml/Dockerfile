FROM python:3.12-slim

WORKDIR /app

COPY ml/requirements.txt ./ml/requirements.txt
RUN pip install --no-cache-dir -r ml/requirements.txt

COPY ml/ ./ml/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "ml.predict:app", "--host", "0.0.0.0", "--port", "8000"]
