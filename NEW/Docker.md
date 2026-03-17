## 简单讲解
**Docker** 是一种开源的应用容器引擎。如果把你的应用程序比作“货物”，那么 Docker 就是那个标准化的**集装箱**。  

解决的就是： “**在我电脑上运行得好好的，怎么到你那儿就崩了？**”  



要理解 Docker，必须掌握这三个核心名词：

+ **镜像 (Image)：** 相当于一个“只读模板”。它包含了运行某个软件所需的所有代码、库、环境变量和配置文件。你可以把它看作是安装系统的 ISO 镜像。
+ **容器 (Container)：** 镜像运行时的实体。镜像就像是类，而容器就是实例。容器可以被启动、开始、停止、删除，每个容器之间是相互隔离的。
+ **仓库 (Repository/Registry)：** 集中存放镜像文件的地方。最著名的就是官方的 **Docker Hub**。

一个镜像可以开多个实例，需要先在一台同样架构的电脑上配置镜像然后拉取成tar文件以布置到生产上

1）轻量和高效 

传统的虚拟机（VM）需要运行一个完整的操作系统，而 Docker 容器直接运行在宿主机的内核上，不需要额外的 OS 开销。

+ **虚拟机：** 启动分钟级，占用 GB 级的内存。
+ **Docker：** 启动秒级，占用 MB 级的内存。

2）资源隔离  ：

 Docker 利用 Linux 内核的 `Namespaces` 和 `Control Groups (cgroups)` 技术，确保容器 A 里的程序不会干扰到容器 B。即使一个容器宕机了，也不会影响其他容器。  



 Docker vs Kubernetes (K8s)  ：

+ **Docker** 负责**创建和运行**单个容器。
+ **Kubernetes** 负责**管理和调度**成千上万个容器（比如自动扩容、负载均衡）。

就像 Docker 是一个个集装箱，而 Kubernetes 就是指挥塔和起重机。



<font style="color:rgb(51, 51, 51);">Docker 采用分层存储机制：</font>

<font style="color:rgb(51, 51, 51);">每个镜像由多层组成（如 Ubuntu 基础层 + 应用层）</font>

<font style="color:rgb(51, 51, 51);">相同基础层（如 Ubuntu）可以被多个镜像共享</font>

<font style="color:rgb(51, 51, 51);">拉取 5 个基于 Ubuntu 的镜像，只会存储一份 Ubuntu 基础层</font>



<font style="color:rgb(51, 51, 51);">Dockerfile 和 Docker Compose 有什么区别？</font>

**<font style="color:rgb(51, 51, 51);">Dockerfile</font>**<font style="color:rgb(51, 51, 51);"> = 你的"建房图纸"（定义房子怎么建）docker build</font>

**<font style="color:rgb(51, 51, 51);">Docker Compose</font>**<font style="color:rgb(51, 51, 51);"> = 你的"小区规划图"（定义小区里房子怎么安排）docker compose up</font>

### <font style="color:rgb(51, 51, 51);">docker基础原理：</font>
| **<font style="color:rgb(51, 51, 51);">组件</font>** | **<font style="color:rgb(51, 51, 51);">说明</font>** | **<font style="color:rgb(51, 51, 51);">作用</font>** |
| :--- | :--- | :--- |
| **<font style="color:rgb(51, 51, 51);">Docker Daemon</font>** | <font style="color:rgb(51, 51, 51);">后台守护进程</font> | <font style="color:rgb(51, 51, 51);">管理容器生命周期</font> |
| **<font style="color:rgb(51, 51, 51);">Docker Client</font>** | <font style="color:rgb(51, 51, 51);">用户交互接口</font> | <font style="color:rgb(51, 51, 51);">发送命令给Daemon</font> |
| **<font style="color:rgb(51, 51, 51);">REST API</font>** | <font style="color:rgb(51, 51, 51);">客户端与Daemon通信</font> | <font style="color:rgb(51, 51, 51);">标准化接口</font> |
| **<font style="color:rgb(51, 51, 51);">Docker Image</font>** | <font style="color:rgb(51, 51, 51);">只读模板</font> | <font style="color:rgb(51, 51, 51);">定义应用及其运行环境</font> |
| **<font style="color:rgb(51, 51, 51);">Docker Container</font>** | <font style="color:rgb(51, 51, 51);">镜像运行实例</font> | <font style="color:rgb(51, 51, 51);">独立、轻量、高效的运行单元</font> |
| **<font style="color:rgb(51, 51, 51);">Docker Registry</font>** | <font style="color:rgb(51, 51, 51);">镜像存储平台</font> | <font style="color:rgb(51, 51, 51);">如Docker Hub，用于存储和分发镜像</font> |


### 技术实现原理
<font style="color:rgb(51, 51, 51);">Linux实现</font>

<font style="color:rgb(51, 51, 51);">基于命名空间（Namespaces）</font>**<font style="color:rgb(51, 51, 51);">和</font>**<font style="color:rgb(51, 51, 51);">控制组（Cgroups）</font>

+ <font style="color:rgb(51, 51, 51);">命名空间：实现隔离（进程、网络、文件系统等）</font>
+ <font style="color:rgb(51, 51, 51);">Cgroups：限制和管理资源（CPU、内存等）</font>
+ <font style="color:rgb(51, 51, 51);">联合文件系统（如OverlayFS）：实现镜像分层存储</font>

<font style="color:rgb(51, 51, 51);">Windows实现</font>

+ <font style="color:rgb(51, 51, 51);">Windows容器模式：直接运行Windows容器（共享内核）</font>
+ <font style="color:rgb(51, 51, 51);">Linux容器模式：通过Hyper-V运行Linux内核（Moby VM）</font>

### <font style="color:rgb(51, 51, 51);">Docker端口映射</font>
<font style="color:rgb(51, 51, 51);">自动映射：docker run -d -P image</font>

<font style="color:rgb(51, 51, 51);">指定映射：docker run -d -p 8080:80 image 生产环境，需要固定端口</font>

<font style="color:rgb(51, 51, 51);">EXPOSE映射：docker run -d -P image 需要与Dockerfile配合</font>

<font style="color:rgb(51, 51, 51);">端口冲突检查：</font>

### <font style="color:rgb(51, 51, 51);">挂载卷</font>
| **<font style="color:rgb(51, 51, 51);">类型</font>** | **<font style="color:rgb(51, 51, 51);">特点</font>** | **<font style="color:rgb(51, 51, 51);">适用场景</font>** | **<font style="color:rgb(51, 51, 51);">优点</font>** | **<font style="color:rgb(51, 51, 51);">缺点</font>** |
| :--- | :--- | :--- | :--- | :--- |
| **<font style="color:rgb(51, 51, 51);">绑定挂载（Bind Mounts）</font>** | <font style="color:rgb(51, 51, 51);">挂载主机任意路径</font> | <font style="color:rgb(51, 51, 51);">本地开发、配置文件</font> | <font style="color:rgb(51, 51, 51);">无需额外管理，性能高</font> | <font style="color:rgb(51, 51, 51);">依赖主机文件结构</font> |
| **<font style="color:rgb(51, 51, 51);">Docker卷（Volumes）</font>** | <font style="color:rgb(51, 51, 51);">Docker管理的特定路径</font> | <font style="color:rgb(51, 51, 51);">数据持久化、容器间共享</font> | <font style="color:rgb(51, 51, 51);">易于管理，独立于主机</font> | <font style="color:rgb(51, 51, 51);">需要额外命令创建</font> |


### <font style="color:rgb(51, 51, 51);">docker存储的三层级</font>
<font style="color:rgb(51, 51, 51);">第1层：</font>**<font style="color:rgb(51, 51, 51);">镜像（Image）</font>**<font style="color:rgb(51, 51, 51);"> ← </font>**<font style="color:rgb(51, 51, 51);">只读，永久存储</font>**

<font style="color:rgb(51, 51, 51);">基础系统+python环境 </font><font style="color:rgb(51, 51, 51);">+</font><font style="color:rgb(51, 51, 51);"> 预装的包</font>

<font style="color:rgb(51, 51, 51);">第2层：</font>**<font style="color:rgb(51, 51, 51);">容器（Container）</font>**<font style="color:rgb(51, 51, 51);"> ← </font>**<font style="color:rgb(51, 51, 51);">可写层，临时存储</font>**

<font style="color:rgb(51, 51, 51);">基于镜像创建，包含：运行时文件 + 下载的依赖 + 临时数据 </font>**<font style="color:rgb(51, 51, 51);">容器删除 = 所有数据丢失</font>**

`**<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">--force-recreate</font>**`**<font style="color:rgb(51, 51, 51);"> 会删除这个层</font>**

<font style="color:rgb(51, 51, 51);">第3层：</font>**<font style="color:rgb(51, 51, 51);">卷（Volume）</font>**<font style="color:rgb(51, 51, 51);"> ← </font>**<font style="color:rgb(51, 51, 51);">持久化存储</font>**

<font style="color:rgb(51, 51, 51);">通过 </font>`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">volumes</font>`<font style="color:rgb(51, 51, 51);"> 挂载到容器</font>

**<font style="color:rgb(51, 51, 51);">独立于容器生命周期</font>**<font style="color:rgb(51, 51, 51);">，容器删除后依然存在</font>

## 部署和使用：
一个典型的 Docker 工作流通常分为三步：

1. **Build（构建）：** 编写一个 `Dockerfile`（自动化脚本），定义你的应用环境。
2. **Ship（运输）：** 将构建好的镜像推送到镜像仓库。（这里也可呀本地到处tar）
3. **Run（运行）：** 在任何安装了 Docker 的机器上拉取镜像并运行。

```python
# 一个简单的 Dockerfile 例子
FROM python:3.9               # 基础镜像
WORKDIR /app                  # 设置工作目录
COPY . .                      # 复制代码
RUN pip install -r requirements.txt  # 安装依赖
CMD ["python", "app.py"]      # 运行命令

或者在已有镜像上新增：
FROM algo_base:1.1
ENV TIME_ZONE=Asia/Shanghai

WORKDIR /workspace/intrusion_detection
COPY ./code /workspace/intrusion_detection
# COPY msyh.ttc /root/.config/Ultralytics/msyh.ttc

CMD ["python", "intrusion_app_mq.py"]

## docker build -t intrusion_detection:1.0 .
```

## docker常用指令与运维
```python
docker构建镜像：
docker build -t ragflow:local .

查看镜像
docker images

删除容器
docker rm -f 容器名

删除镜像
docker rmi -f 镜像名

启动容器
docker run

停止容器
docker stop

查看所有包括启动的容器
docker ps -a 

进入容器中
docker exec -it ID /bin/bash

彻底清除缓存
docker compose down

docker compose up -d --force-recreate

进入容器校验
docker exec -it docker-ragflow-cpu-1 /bin/bash

sudo -i 进入root用户

临时调试
docker run -it(-it进入容器) --rm alpine（停止后删除临时调试容器）

自动重启
sudo docker run -d --restart always 加-d就是挂后台不用占用窗口


查看日志
docker logs -f

sudo docker run -d --restart always 总是重启 只要 Docker 服务在运行，该容器就必须运行
即便你手动执行了 docker stop 停止了它，只要 Docker 引擎重启（比如你重启了电脑或 WSL）
，这个容器依然会自动启动。

sudo docker run -d --restart unless-stopped
除非手动停止 如果你手动执行了 docker stop algorithm，那么在你重启电脑或重启 Docker 引擎后，
这个容器不会自动启动。它会保持停止状态，直到你手动执行 docker start

docker image prune 清理空间

查看日志运维
docker logs --since 5m <容器名称或ID>
docker logs --since "2026-03-17T23:00:00"
docker logs --tail 100 <容器名称或ID>
docker logs -f --tail 100 <容器名称或ID>
docker logs --since 30m algorithm | grep "Error"
docker logs algorithm > debug.log

tail -f logs/alarm_video_select_server.log
ps -ef | grep mc_demo_yolov7
ps -ef | grep intrusion_app_mq.py

```

