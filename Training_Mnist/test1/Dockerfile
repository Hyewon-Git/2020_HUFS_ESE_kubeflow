

FROM tensorflow/tensorflow:2.1.0-py3

RUN mkdir -p /app
ADD step1-katib-mnist-job.py /app/

ENTRYPOINT ["python", "/app/step1-katib-mnist-job.py"]
