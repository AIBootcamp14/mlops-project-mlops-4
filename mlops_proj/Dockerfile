FROM python:3.11-bookworm

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

COPY dataset /root/dataset
COPY src /root/src
COPY start_api_server.sh /root

WORKDIR /root
