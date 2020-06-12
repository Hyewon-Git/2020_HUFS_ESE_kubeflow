FROM tensorflow/tensorflow:2.1.0-py3

RUN mkdir -p /app
ADD katib-mnist-random-job2.py /app/

ENTRYPOINT ["python", "/app/katib-mnist-random-job2.py"]
