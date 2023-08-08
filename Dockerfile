FROM registry-svc:25000/library/ubuntu_py3.9.8_tf2.5.0_cuda11:v1.0.3


# 复制文件
RUN mkdir /ntt
RUN mkdir /ntt/alphamind
WORKDIR /opt
ADD ./datasets ./datasets
ADD ./log ./log
ADD ./utils ./utils
ADD ./config ./config
ADD .env .


ADD ./requirements.txt .
RUN pip install -r requirements.txt -i http://134.80.159.26:8080/simple --trusted-host 134.80.159.26  


ADD flasktest.py .
ADD release.sh .

ADD server.py .
ADD server.sh .

ADD train.py .
ADD train.sh .

