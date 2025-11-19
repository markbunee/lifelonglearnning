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

### 2.1以ragflow部署为例

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

复用容器：

docker compose up -d

强制重启容器：

docker compose up -d --force-recreate

docker存储的三层级

第1层：**镜像（Image）** ← **只读，永久存储**

基础系统+python环境 \+ 预装的包

第2层：**容器（Container）** ← **可写层，临时存储**

基于镜像创建，包含：运行时文件 + 下载的依赖 + 临时数据 **容器删除 = 所有数据丢失**

**`--force-recreate` 会删除这个层**

 第3层：**卷（Volume）** ← **持久化存储**

通过 `volumes` 挂载到容器

**独立于容器生命周期**，容器删除后依然存在

**目前只挂载了代码和日志，没挂载缓存目录**

```
┌─────────────────┐
│     Volume      │ ← 持久化（你的D盘）
│  /ragflow       │ ← 代码
│  /ragflow/logs  │ ← 日志  
└─────────────────┘
         ↑
         │ 挂载
┌─────────────────┐
│    Container    │ ← 临时（每次 --force-recreate 都重置）
│  /root/.cache/uv│ ← 依赖缓存（丢失！）
│  /tmp           │ ← 临时文件（丢失！）
└─────────────────┘
         ↑
         │ 基于
┌─────────────────┐
│      Image      │ ← 只读（不变）
│  Ubuntu + Python│
│  基础环境        │
└─────────────────┘
```

| 你的说法                             | 是否正确   | 说明                                                         |
| ------------------------------------ | ---------- | ------------------------------------------------------------ |
| "官方镜像所有环境和代码都在镜像本身" | ❌ **错误** | 镜像只含基础环境，**应用依赖需启动时下载**（RAGFlow设计如此） |
| "挂载代码 = 加载本地代码"            | ✅ **正确** | 你挂载了 `/ragflow`，容器内代码来自你的本地文件              |
| "未挂载缓存 = 每次启动重下"          | ✅ **正确** | 未挂载缓存目录 → 依赖在容器临时层 → 容器重建就丢失           |
| "必须创建缓存文件夹才能持久化"       | ✅ **正确** | Docker **不会自动创建**目录，必须手动 `mkdir -p uv-cache`    |
| "创建过程中缓存会存到其他地方"       | ❌ **错误** | 未挂载缓存时，依赖**只存在容器临时层**（删除容器=消失），**不会残留宿主机** |

所以首先我如果完全按照官方拉取ragflow的镜像，那基础环境（无依赖）和代码都是镜像本身有的，然后我现在挂载出来那我就是加载我挂载的代码和环境缓存，如果我没有挂载环境缓存，那么每一次启动容器都要重新下载一次吗，然后是否在创建过程中有缓存或者下载到其他地方的垃圾呢，然后还有我是否一定要创建环境缓存的文件夹，这样才能下载下来而不是又只是暂存，以上我理解的对吗

| 特性维度       | 直接拉取镜像 (Docker Compose) | 卷挂载开发 (混合模式)  | 源码部署 (完整开发)  |
| :------------- | :---------------------------- | :--------------------- | :------------------- |
| **目标用户**   | 最终用户、测试人员、生产部署  | 二次开发者、定制化需求 | 核心贡献者、深度定制 |
| **技术难度**   | ⭐☆☆☆☆ (非常简单)              | ⭐⭐☆☆☆ (中等)           | ⭐⭐⭐⭐⭐ (复杂)         |
| **启动速度**   | ⭐⭐⭐⭐⭐ (几分钟)                | ⭐⭐⭐⭐☆ (中等)           | ⭐⭐☆☆☆ (较慢)         |
| **代码修改**   | 不支持                        | ✅ 实时热更新           | ✅ 完整调试支持       |
| **调试便利性** | 仅日志调试                    | 有限调试               | ⭐⭐⭐⭐⭐ (IDE断点)      |
| **环境依赖性** | 只需Docker                    | 需要Docker+代码        | 完整开发环境         |
| **灵活性**     | 固定功能                      | 中等定制               | 完全自由定制         |
| **适用场景**   | 生产部署、快速体验            | 功能修改、界面定制     | 架构改动、新功能开发 |
