FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app
COPY infer.py /app
COPY train.py /app
COPY model/cifar_net.pth /app/model/

CMD ["python", "app.py"]
