FROM ubuntu

USER root

COPY . .

RUN apt-get update && \
    apt-get install -y python3-pip python3 tesseract-ocr libtesseract-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages mede

CMD ["bash"]
