FROM python:3.10-slim
WORKDIR /src
COPY . /src/
RUN pip install --no-cache-dir -r /src/requirements.txt
CMD ["python", "/src/main.py"]