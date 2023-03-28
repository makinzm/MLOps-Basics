FROM python:3.8

WORKDIR /root/working

RUN apt-get update && \
    apt-get install -y python3-dev python3-pip python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --no-cache-dir jupyter notebook && pip install --no-cache-dir -r requirements.txt

RUN pip3 uninstall transformers datasets -y &&\
    pip3 install transformers datasets -q

WORKDIR /root

CMD ["bash"]
