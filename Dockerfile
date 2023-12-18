FROM python:3.9

WORKDIR /code
ENV PATH=$PATH:/code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY serve .
COPY ./app /code/app

EXPOSE 8080
RUN chmod +x serve
