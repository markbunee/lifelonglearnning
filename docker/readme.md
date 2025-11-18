# DOCKER

## 1.资料



## 2.笔记

![image-20250702160224236](./pic/image-20250702160224236.png)

![image-20250702160259835](./pic/image-20250702160259835.png)

![image-20250702160329463](./pic/image-20250702160329463.png)

![image-20250702160419735](./pic/image-20250702160419735.png)

![image-20250702160533875](./pic/image-20250702160533875.png)

docker pull nginx

sudo docker pull nginx

sudo vi /etc/docker/daemon.json

sudo service docker restart

sudo docker pull nginx

dicjer images

sudo docker rm

QEMU指令模拟

sudo docker run nginx

sudo docker ps

删除容器时docker rm -f 

删除镜像是docker rmi <image-id>

临时调试一个容器的命令exit之后就立即删除

sudo docker run -it –rm alpine 

sudo docker run -d —restart unless

docker run 

docker stop 

docker ps -a 查看所有包括启动的容器

sudo docker inspect 

docker create -p 

docker logs  

sudo docker exec c29a ps -ef

docker exec -it ID /bin/bash

docker 内部是极简的操作系统所以很多都是缺失的

Dockerfile文件固定的

开头都是

FROM python:3.13-slim

WORKDIR /app

COPY .  .

RUN pip install -r requiremnets.txt

docker build -t docker_test .

.代表再本地

docker run -d -p 8000:8000 docker_test

docker login 

推送镜像到dockerhub

docker push yourname/docker_test

docker pull 

docker network create network1

![image-20251014144008572](./pic/image-20251014144008572.png)

podman pull docker.io/library/mongo 

podman不支持简略名要给到dockerhub国内需要改配置文件

sudo vi /etc/containers/registries.conf更改配置文件

-e 传递环境变量

ip dder pod

![image-20251014144639432](./pic/image-20251014144639432.png)

与k8s无缝衔接并且借助配置文件替代原来dockers的compose

Docker 采用分层存储机制

每个镜像由多层组成（如 Ubuntu 基础层 + 应用层）

相同基础层（如 Ubuntu）可以被多个镜像共享

拉取 5 个基于 Ubuntu 的镜像，只会存储一份 Ubuntu 基础层

### docker的挂载

Dockerfile 和 Docker Compose 有什么区别？

**Dockerfile** = 你的"建房图纸"（定义房子怎么建）docker build

**Docker Compose** = 你的"小区规划图"（定义小区里房子怎么安排）docker compose up

docker compose up -d --force-recreate

彻底清除缓存：docker compose down

docker ps 看映射

