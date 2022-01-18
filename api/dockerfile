FROM python:3.9
WORKDIR /code
COPY ./requirements.dev.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY outputs /code/outputs
COPY main.py main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]