 # Dockerfile, Image, Container


FROM python:3.9.7-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt


RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6 libgl1 -y

COPY . /app

EXPOSE 8000

#  ENTRYPOINT ["streamlit", "run"]

#  CMD ["./app/app.py"]


CMD streamlit run ./app/app.py --server.port $PORT

