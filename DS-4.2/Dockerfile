FROM python:3.8

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY dags/config.ini /app/config.ini

CMD [ "python", "./dags/pipepline_dag.py" ]

