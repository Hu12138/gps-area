FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY main.py .

RUN pip install --no-cache-dir -r requirements.txt

# 设置默认环境变量（可在 docker run 时覆盖）
ENV GUNICORN_WORKERS=4

EXPOSE 5000

# 直接用环境变量启动 gunicorn
CMD gunicorn -w ${GUNICORN_WORKERS} -b 0.0.0.0:5000 main:app