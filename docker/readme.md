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

进入到容器内验证：

docker exec -it docker-ragflow-cpu-1 bash

ls -la /ragflow/



docker基础原理：

核心组件：

| 组件                 | 说明               | 作用                             |
| -------------------- | ------------------ | -------------------------------- |
| **Docker Daemon**    | 后台守护进程       | 管理容器生命周期                 |
| **Docker Client**    | 用户交互接口       | 发送命令给Daemon                 |
| **REST API**         | 客户端与Daemon通信 | 标准化接口                       |
| **Docker Image**     | 只读模板           | 定义应用及其运行环境             |
| **Docker Container** | 镜像运行实例       | 独立、轻量、高效的运行单元       |
| **Docker Registry**  | 镜像存储平台       | 如Docker Hub，用于存储和分发镜像 |

### 技术实现原理

Linux实现

基于命名空间（Namespaces）**和**控制组（Cgroups）

- 命名空间：实现隔离（进程、网络、文件系统等）
- Cgroups：限制和管理资源（CPU、内存等）
- 联合文件系统（如OverlayFS）：实现镜像分层存储

Windows实现

- Windows容器模式：直接运行Windows容器（共享内核）
- Linux容器模式：通过Hyper-V运行Linux内核（Moby VM）

Docker端口映射

自动映射：docker run -d -P image

指定映射：docker run -d -p 8080:80 image  生产环境，需要固定端口

EXPOSE映射：docker run -d -P image 需要与Dockerfile配合

端口冲突检查：

卷：

| 类型                        | 特点                 | 适用场景               | 优点                 | 缺点             |
| --------------------------- | -------------------- | ---------------------- | -------------------- | ---------------- |
| **绑定挂载（Bind Mounts）** | 挂载主机任意路径     | 本地开发、配置文件     | 无需额外管理，性能高 | 依赖主机文件结构 |
| **Docker卷（Volumes）**     | Docker管理的特定路径 | 数据持久化、容器间共享 | 易于管理，独立于主机 | 需要额外命令创建 |

| 项目         | 未挂载配置               | 你的配置                      |
| ------------ | ------------------------ | ----------------------------- |
| **代码修改** | 需重建镜像+重启容器      | **直接修改本地文件→立即生效** |
| **文档存储** | 临时存储，容器删除即丢失 | 永久保存在本地D盘             |
| **配置管理** | 需进入容器修改           | 直接编辑本地文件              |
| **开发效率** | 低（10+秒/次）           | **高（实时生效）**            |

未挂载配置：

| 你的操作         | 实际发生                                     | 为什么                   |
| ---------------- | -------------------------------------------- | ------------------------ |
| **拉取镜像**     | `infiniflow/ragflow:v0.22.0`（包含所有代码） | 镜像 = 代码的“快照”      |
| **启动容器**     | 容器运行镜像中的代码（不读取本地文件）       | 代码在容器内部，不在本地 |
| **修改本地代码** | ❌ **容器完全无视**                           | 容器没“看到”你的本地文件 |

```
Docker 镜像 (infiniflow/ragflow:v0.22.0)
│
├── ragflow/          ← 代码（固定在镜像里）
│   ├── api.py
│   └── ...
│
└── ... (其他依赖)

↓ 启动容器

容器内部 (/ragflow/)
│
├── ragflow/          ← **镜像里的代码（不可变）**
│   ├── api.py        ← 你改本地的 api.py 也没用！
│   └── ...
│
└── ... (运行时数据)
```

容器启动后，**镜像内容被冻结**，除非你重新构建镜像，否则容器永远用镜像里的代码

原始配置下如何更新代码：

**重新构建镜像**

```
docker build -t ragflow:local .
```

**重启容器**（用新镜像）

```
docker compose up -d --force-recreate
```

挂载本地后：

```
本地目录 (D:\ASUS\ragflow-0.22.0\ragflow\)
│
├── api.py            ← 你修改的文件
│
↓ 挂载到容器

容器内部 (/ragflow/ragflow/)
│
├── api.py            ← **实时同步！** 修改本地 = 修改容器
│
└── ...               ← 代码被容器直接读取
```

docker镜像包含：

**完整的环境**：包括操作系统基础层、依赖库、运行时环境

**初始代码**：镜像构建时包含的代码（不是你本地的代码！）

**静态文件**：预编译的资源、配置文件等

测试是否挂载本地

```
docker compose up -d

# D:\ASUS\ragflow-0.22.0\ragflow\api.py
def health_check():
    return {"status": "ok", "local_code": "MODIFIED!"}  # 添加这行
    
curl http://localhost:1234/api/health

{"status": "ok", "local_code": "MODIFIED!"}
```
