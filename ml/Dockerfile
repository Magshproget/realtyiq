FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ml/requirements.txt ./ml/requirements.txt
RUN pip install --no-cache-dir -r ml/requirements.txt

COPY ml/predict.py ./ml/predict.py

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "ml.predict:app", "--host", "0.0.0.0", "--port", "8000"]
