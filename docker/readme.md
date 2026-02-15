# DOCKER及项目部署与运维

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

docker rm -f 删除容器
docker rmi -f 删除镜像
docker volume inspect nginx 挂载卷命令
sudo -i 进入root用户
docker run -it(-it进入容器) --rm alpine（停止后删除临时调试容器）
sudo docker run -d --restart always
sudo docker run -d --restart unless-stopped
docker run 创建并运行
docker create docker start
docker ps -a 加a看到停止
docker stop和start会记录信息
docker logs -f
容器两大技术本质来源于linux
Namespaces 的作用是环境隔离
进程隔离网络隔离用户权限隔离
controlgroups 资源管理
负责资源调度和管理
虚拟机是模拟物理机 容器就是宿主机上的角色扮演

docker exec liux命令

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

启动镜像：

在有dockerfile的项目跟目录下执行：

docker build -t diarization-api:latest .

docker run --rm --gpus all -p 8123:8123 -v %cd%\results:/app/results diarization-api:latest

## 水务：

docker run -d --gpus all  -v /data/suanfa/suanfa0918:/home/bamb/brake/demo/images/upload  -p 6000:6000  --name algorithm  algorithm_image

cd /data/suanfa/suanfa0918

cd /home/bamb/brake/demo/images/upload/img/aiimg

docker inspect 88a720b7d42a  | grep -A 10 -B 5 Mounts

docker run -d --gpus all  -v /data/suanfa/suanfa0918:/home/bamb/brake/demo/images/upload  -p 6000:6000  --name algorithm_image  algorithm_imadge

docker ps

docker logs algorithm

mkdir -p /data/suanfa/suanfa0918/img/aiimg

docker stop 

docker_container

    {
        "aiParam": {
            "algCodes": [
                "B"
            ],
            "file":"1.png",
            "businessId": "1892824983833292800",
           "geoCameraParam": {
                "boundaryParam": [
                    {
                        "boundaryPoint": [
                            {
                                "lon": 111.49056470100004,
                                "lat": 21.762450872000045
                            },
                            {
                                "lon": 111.49063427800007,
                                "lat": 21.76251000800005
                            },
                            {
                                "lon": 111.49069295900006,
                                "lat": 21.76253750500007
                            },
                            {
                                "lon": 111.49075401000005,
                                "lat": 21.76253732400005
                            },
                            {
                                "lon": 111.49085216400005,
                                "lat": 21.76255905000005
                            },
                            {
                                "lon": 111.49092671400007,
                                "lat": 21.762511118000077
                            },
                            {
                                "lon": 111.49120446200004,
                                "lat": 21.762388748000035
                            },
                            {
                                "lon": 111.49074621500006,
                                "lat": 21.761819178000053
                            },
                            {
                                "lon": 111.49050878200006,
                                "lat": 21.761935694000044
                            },
                            {
                                "lon": 111.49047486600006,
                                "lat": 21.761943833000032
                            },
                            {
                                "lon": 111.49044147800004,
                                "lat": 21.76196228300006
                            },
                            {
                                "lon": 111.49038120600005,
                                "lat": 21.762001358000077
                            },
                            {
                                "lon": 111.49035059000005,
                                "lat": 21.762053060000028
                            },
                            {
                                "lon": 111.49035073700009,
                                "lat": 21.762096427000074
                            },
                            {
                                "lon": 111.49037522600008,
                                "lat": 21.76217039900007
                            },
                            {
                                "lon": 111.49038427000005,
                                "lat": 21.762234438000064
                            },
                            {
                                "lon": 111.49040466700006,
                                "lat": 21.76229644700004
                            },
                            {
                                "lon": 111.49047462700008,
                                "lat": 21.76236016100006
                            },
                            {
                                "lon": 111.49052000500006,
                                "lat": 21.76239646700003
                            },
                            {
                                "lon": 111.49056470100004,
                                "lat": 21.762450872000045
                            }
                        ],
                        "region": "f4d0d4551221f8ca5c1ce95bd85184c7-0"
                    }
                ],
    
                "minLon": 111.488432,
                "maxLon": 111.493268,
                "minLat": 21.760778,
                "maxLat": 21.764372
            }
        },
        "aiType": "1",
        "bussParam": {
            "reqId": "1",
            "taskId": "1001",
            "taskNo": "No1002"
        },
        "resType": 1,
        "callbackUrl": "http://host.docker.internal:/{businessId}"
    }

curl -X POST "http://localhost:6000" \
  -H "Content-Type: application/json" \
  -d '{
    "aiParam": {
        "algCodes": ["B"],
        "file": "text.png",
        "businessId": "1892824983833292800",
        "geoCameraParam": {
            "boundaryParam": [{
                "boundaryPoint": [
                    {"lon": 111.49056470100004, "lat": 21.762450872000045},
                    {"lon": 111.49063427800007, "lat": 21.76251000800005},
                    {"lon": 111.49069295900006, "lat": 21.76253750500007},
                    {"lon": 111.49075401000005, "lat": 21.76253732400005},
                    {"lon": 111.49085216400005, "lat": 21.76255905000005},
                    {"lon": 111.49092671400007, "lat": 21.762511118000077},
                    {"lon": 111.49120446200004, "lat": 21.762388748000035},
                    {"lon": 111.49074621500006, "lat": 21.761819178000053},
                    {"lon": 111.49050878200006, "lat": 21.761935694000044},
                    {"lon": 111.49047486600006, "lat": 21.761943833000032},
                    {"lon": 111.49044147800004, "lat": 21.76196228300006},
                    {"lon": 111.49038120600005, "lat": 21.762001358000077},
                    {"lon": 111.49035059000005, "lat": 21.762053060000028},
                    {"lon": 111.49035073700009, "lat": 21.762096427000074},
                    {"lon": 111.49037522600008, "lat": 21.76217039900007},
                    {"lon": 111.49038427000005, "lat": 21.762234438000064},
                    {"lon": 111.49040466700006, "lat": 21.76229644700004},
                    {"lon": 111.49047462700008, "lat": 21.76236016100006},
                    {"lon": 111.49052000500006, "lat": 21.76239646700003},
                    {"lon": 111.49056470100004, "lat": 21.762450872000045}
                ],
                "region": "f4d0d4551221f8ca5c1ce95bd85184c7-0"
            }],
            "minLon": 111.488432,
            "maxLon": 111.493268,
            "minLat": 21.760778,
            "maxLat": 21.764372
        }
    },
    "aiType": "1",
    "bussParam": {
        "reqId": "1",
        "taskId": "1001",
        "taskNo": "No1002"
    },
    "resType": 1,
    "callbackUrl": "http://host.docker.internal:8999/1892824983833292800"
}

/api/predict

/analyze

**`-H`**：curl命令中设置HTTP头部的参数

**`Content-Type`**：HTTP头部字段，描述请求体的数据类型

**`application/json`**：表示数据格式是JSON



### 常见的AI服务API端点模式：

#### 1. **直接动词型**（最常用）

- `/predict`- 预测（机器学习模型最常用）
- `/analyze`- 分析
- `/detect`- 检测
- `/classify`- 分类

#### 2. **API版本化**

- `/api/predict`
- `/v1/predict`
- `/v1/api/predict`

#### 3. **模型服务型**

- `/model/predict`
- `/inference`
- `/detection`

docker logs algorithm_container

```
curl -X POST "http://localhost:18080/api/analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "aiParam": {
        "algCodes": ["B"],
        "file": "/home/bamb/brake/demo/images/upload/img/aiimg/1764145638549_test.png",
        "businessId": "1892824983833292800",
        "geoCameraParam": {
            "boundaryParam": [{
                "boundaryPoint": [
                    {"lon": 111.49056470100004, "lat": 21.762450872000045},
                    {"lon": 111.49063427800007, "lat": 21.76251000800005},
                    {"lon": 111.49069295900006, "lat": 21.76253750500007},
                    {"lon": 111.49075401000005, "lat": 21.76253732400005},
                    {"lon": 111.49085216400005, "lat": 21.76255905000005},
                    {"lon": 111.49092671400007, "lat": 21.762511118000077},
                    {"lon": 111.49120446200004, "lat": 21.762388748000035},
                    {"lon": 111.49074621500006, "lat": 21.761819178000053},
                    {"lon": 111.49050878200006, "lat": 21.761935694000044},
                    {"lon": 111.49047486600006, "lat": 21.761943833000032},
                    {"lon": 111.49044147800004, "lat": 21.76196228300006},
                    {"lon": 111.49038120600005, "lat": 21.762001358000077},
                    {"lon": 111.49035059000005, "lat": 21.762053060000028},
                    {"lon": 111.49035073700009, "lat": 21.762096427000074},
                    {"lon": 111.49037522600008, "lat": 21.76217039900007},
                    {"lon": 111.49038427000005, "lat": 21.762234438000064},
                    {"lon": 111.49040466700006, "lat": 21.76229644700004},
                    {"lon": 111.49047462700008, "lat": 21.76236016100006},
                    {"lon": 111.49052000500006, "lat": 21.76239646700003},
                    {"lon": 111.49056470100004, "lat": 21.762450872000045}
                ],
                "region": "f4d0d4551221f8ca5c1ce95bd85184c7-0"
            }],
            "minLon": 111.488432,
            "maxLon": 111.493268,
            "minLat": 21.760778,
            "maxLat": 21.764372
        }
    },
    "aiType": "1",
    "bussParam": {
        "reqId": "1",
        "taskId": "1001",
        "taskNo": "No1002"
    },
    "resType": 1,
    "callbackUrl": "http://host.docker.internal:8999/1892824983833292800"
}'
```

docker logs Beyondrange

docker stop algorithm 

docker rm algorithm

docker network create app-network

docker run -d --gpus all   -v /data/suanfa/suanfa0918:/home/bamb/brake/demo/images/upload  --network detect-net   --name algorithm  algorithm_image



docker run -d --gpus all   -v /data/suanfa/suanfa0918:/home/bamb  --network detect-net   --name algorithm  algorithm_image



docker run -d  -p 18080:18080 --network detect-net --name Beyondrange  detect-server



/home/bamb/brake/demo/images/upload/img/aiimg/1764145638549_test.png

### 人声分离：

在环境非默认账户情况下激活环境方法：

source /data/lihongbiao/anaconda3/bin/activate diarization



**本地直接运行**

```bash
uvicorn server:app --host 0.0.0.0 --port 8123
# 或启用自动重载：
python server.py --host 0.0.0.0 --port 8111 --reload
```

确保已在当前环境安装 `requirements.txt` 中的依赖，并准备好模型运行所需的 GPU/CPU。

```bash
pip install cython
conda install ffmpeg
pip install -r requirements.txt
pip install git+https://github.com/MahmoudAshraf97/demucs.git
pip install git+https://github.com/oliverguhr/deepmultilingualpunctuation.git
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
pip install python-multipart
pip install torchcodec
```

接口示例（与 `server.py` 定义一致）：

```bash
curl -X POST "http://localhost:8123/diarize" \
  -F "url=http://192.168.30.165/file-resource/1919643053772152833/bz_opinion_analysis_file/analysis/20250811142333-18679590406-S20250811142333431037AC130E1711279928-0000107800184263_1763022114938.mp3" \
  -F "language=zh" \
  -F "whisper_model=small" \
  -F "device=cuda" \
  -F "device_index=1" \
  -F "no_stem=true" \
  

```

可选表单字段说明：

- `whisper_model`：覆盖默认的 Whisper 模型。
- `device`：强制使用 `cuda` 或 `cpu`。
- `no_stem`：设为 `true` 跳过 Demucs 分离。
- `diarize_extra`：用空格分隔的额外参数，会原样传递给 `app.py`（例如 `--vad-strict --batch-size 4`）。

返回结果中会包含处理后的文件路径（本地或容器内的 `results/` 目录），可根据需要进一步读取或下载。***

具体例子

```bash
curl -X POST "http://127.0.0.1:8000/diarize"   \
-F "file=@/home/tommy/diarization-api/diarization/test/test-301.mp3"   \
-F "language=zh"   \
-F "diarize_extra=--whisper-model medium"
```

 conda install -c conda-forge ffmpeg 

conda install -c conda-forge libsndfile 

conda install pytorch==2.1.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install python-multipart

pip install git+https://gitclone.com/github.com/oliverguhr/deepmultilingualpunctuation.git

pip install git+https://gitclone.com/github.com/MahmoudAshraf97/demucs.git

pip install git+https://gitclone.com/github.com/MahmoudAshraf97/ctc-forced-aligner.git

screen -S yuyin

ctrl + A +D临时退出

提供权限

sudo chmod -R 777 /tmp/mxx/new1126/diarization-apiorg

export HF_ENDPOINT=https://hf-mirror.com

export HF_ENDPOINT=https://mirrors.sustech.edu.cn/hugging-face export HF_ENDPOINT=https://hf-mirror.com

```
docker run --rm --gpus all   -e CUDNN_LIB_DIR=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib   -e LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib   -v $(pwd)/results:/app/results   -p 8124:8000    -e HF_ENDPOINT=https://hf-mirror.com   diarization-api
```

```
curl -X POST "http://localhost:8124/diarize"   -F "url=http://192.168.30.165/file-resource/1919643053772152833/bz_opinion_analysis_file/analysis/20250811142333-18679590406-S20250811142333431037AC130E1711279928-0000107800184263_1763022114938.mp3"   -F "language=zh"   -F "whisper_model=medium"   -F "device=cuda"   -F "no_stem=true" 
```

docker run --rm --gpus "device=1" \
  -e CUDNN_LIB_DIR=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib \
  -e LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v $(pwd)/results:/app/results \
  -p 8124:8000 \
  diarization-api





## diarization-api

Whisper + NeMo 的一站式语音分离/转写/角色分类服务。提供：

- `app.py` 命令行：对单个音频跑完整管线，产出 `txt`/`srt`/`csv`。
- `server.py` FastAPI：上传音频即返回结果，带定期清理输出。
- Docker 镜像：封装 CUDA 12.4 运行环境，可 GPU/CPU 运行。

### 项目结构

```
.
├─app.py                # CLI 入口，调用 diarization/diarize.py 并做 SRT 解析、角色分类、CSV 导出
├─server.py             # FastAPI 封装上传/调用/结果缓存与清理
├─Dockerfile            # 基于 pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
├─requirements.txt      # 推理依赖，配合 constraints.txt 固定版本
├─trainer.py            # 示例文本角色分类训练脚本
├─diarization/          # Whisper 转写 + MSDD 说话人分离 + CTC 对齐 + 标点恢复 主流程
├─results/              # 默认输出目录（容器可挂载到宿主）
├─text_role_classifier.pkl 等  # 已训练好的角色分类模型
└─evaluation_manifest.json / metrics_test.py / test_predict_role.py # 测试与数据示例
```

### 核心流程

- 可选音源分离：Demucs 保留人声（`--no-stem` 可关闭）。
- Faster-Whisper 转写，按需指定 `--whisper-model`。
- CTC forced aligner 做字/词级时间对齐，配合标点恢复。
- NeMo MSDD 说话人分离，VAD 可通过 `--vad-*` 调整。
- 生成带说话人分段的 `txt` / `srt`，并用文本分类模型产出角色 CSV。
- FastAPI 封装上传、调用 CLI、汇总输出，并定时清理过期结果。

### 技术栈

- 深度学习/ASR：PyTorch 2.4.1、Faster-Whisper、CTranslate2、NVIDIA NeMo (MSDD)。
- 音频处理：ffmpeg/sox、librosa、pydub、Demucs。
- 对齐与文本：ctc-forced-aligner、deepmultilingualpunctuation、transformers、scikit-learn。
- 服务层：FastAPI + Uvicorn，容器化用官方 PyTorch CUDA Runtime 基础镜像。

### Docker 运行

```bash
# 构建镜像
docker build -t diarization-api .

# GPU 模式（推荐）：记得传入 cuDNN 路径并挂载 results
docker run --rm --gpus all \
  -e CUDNN_LIB_DIR=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib \
  -e LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib \
  -v $(pwd)/results:/app/results \
  -p 8124:8000 \
   -e HF_ENDPOINT=https://hf-mirror.com \
  diarization-api

# 上例将宿主机 9000 端口映射到容器 8000。变量一定要加上，不然无法识别cudnn。

# CPU 模式：清空 CUDA_VISIBLE_DEVICES
docker run --rm -p 8000:8000 -e CUDA_VISIBLE_DEVICES= -v $(pwd)/results:/app/results diarization-api

镜像默认启动 FastAPI，接口同下文示例。若要直接调用 CLI，可覆盖入口：

```bash
docker run --rm --gpus all -v $(pwd):/app diarization-api \
  python app.py --audio /app/your_audio.wav --language zh --whisper-model medium
```

---

**本地直接运行**

确保已在当前环境安装 `requirements.txt` 中的依赖，并准备好模型运行所需的 GPU/CPU。

```bash
pip install cython
sudo apt update && sudo apt install ffmpeg
pip install -c constraints.txt -r requirements.txt
```

运行

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
# 或启用自动重载：
python server.py --host 0.0.0.0 --port 8000 --reload
```


接口示例（与 `server.py` 定义一致）：

```bash
curl -X POST "http://localhost:8000/diarize" \
  -F "file=@/path/to/audio.wav" \
  -F "language=zh" \
  -F "whisper_model=medium" \
  -F "device=cuda" \
  -F "no_stem=true" \
  -F "diarize_extra=--vad-strict"
```

可选表单字段说明：

- `whisper_model`：覆盖默认的 Whisper 模型。
- `device`：强制使用 `cuda` 或 `cpu`。
- `no_stem`：设为 `true` 跳过 Demucs 分离。
- `diarize_extra`：用空格分隔的额外参数，会原样传递给 `app.py`（例如 `--vad-strict --batch-size 4`）。

返回结果中会包含处理后的文件路径（本地或容器内的 `results/` 目录），可根据需要进一步读取或下载。***

具体例子

```bash
curl -X POST "http://127.0.0.1:9000/diarize"   \
-F "file=@/home/tommy/diarization-api/diarization/test/test-301.mp3"   \
-F "language=zh"   \
  -F "diarize_extra=--whisper-model medium" \
  -F "device=cuda"
```

输出说明

- 服务接口返回：源文件名、语言、各类产物路径（`results/` 目录下）以及 CSV 解析后的片段内容。
- 本地/容器内会生成：`*.txt`（逐句转写）、`*.srt`（含说话人分段）、`*_with_roles.csv`（角色与置信度）。
- FastAPI 会每小时清理一次超过 72 小时的 `results/`/`temp_outputs_*` 文件。

docker cp 69a155217130:/root/.cache/huggingface /tmp/mxx/docker-cache/huggingface

docker cp 69a155217130:/root/.cache/torch  /tmp/mxx/docker-cache/torch

export HF_HOME=/mnt/host/d/ASUS/docker-modelcache/huggingface

export HUGGINGFACE_HUB_CACHE=/mnt/host/d/ASUS/docker-modelcache/huggingface/hub

export TRANSFORMERS_CACHE=/mnt/host/d/ASUS/docker-modelcache/huggingface/hub

export TORCH_HOME=/mnt/host/d/ASUS/docker-cache/torch

docker exec -it eaa342445c73 ping -c 3 8.8.8.8

docker exec -it eaa342445c73 ping -c 3 mirrors.aliyun.com

## ragflow 部署

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

| 部署方式             | 作用范围 | 复杂度 | 适用场景   | 输出结果         |
| :------------------- | :------- | :----- | :--------- | :--------------- |
| 源码编译 Docker 镜像 | 构建阶段 | ⭐⭐⭐⭐   | 定制化需求 | 生成 Docker 镜像 |
| 以源代码启动服务     | 开发阶段 | ⭐⭐⭐⭐⭐  | 开发调试   | 运行开发环境     |
| docker compose up    | 部署阶段 | ⭐⭐     | 生产/测试  | 运行完整服务     |

源码编译 Docker 镜像

docker build --platform linux/amd64 -f Dockerfile -t infiniflow/ragflow:nightly .

根据 `Dockerfile`构建一个新的 Docker 镜像   

将源代码、依赖包打包到镜像中   修改了源代码，需要重新打包

生成可重复部署的标准化镜像

以源代码启动服务

不依赖 Docker，直接在宿主机运行

使用本地安装的 Python、Node.js 环境

基础设施（数据库等）仍用 Docker 运行

docker compose up -d

一键部署完整服务

使用**预构建的官方镜像**启动所有服务

包含前端、后端、数据库等完整组件

生产环境就绪的部署方式

| 方面           | 源码编译镜像   | 源码启动服务 | Docker Compose     |
| :------------- | :------------- | :----------- | :----------------- |
| **启动速度**   | 慢（需要构建） | 中等         | **快**（直接运行） |
| **调试便利**   | 困难           | **最容易**   | 困难               |
| **环境一致性** | **最好**       | 依赖本地环境 | **很好**           |
| **资源占用**   | 构建时高       | 运行时中等   | 运行时中等         |
| **适用阶段**   | 构建/发布      | 开发/调试    | 测试/生产          |

**是的，所有三种部署方式最终都消耗本地资源**，只是资源分配和架构不同：

| 部署方式               | 资源消耗位置 | 架构特点                      | 资源效率               |
| :--------------------- | :----------- | :---------------------------- | :--------------------- |
| **Docker直接拉镜像**   | 100%本地资源 | 所有服务在Docker容器内        | 容器有开销，但管理简单 |
| **源码编译Docker镜像** | 100%本地资源 | 自定义镜像+容器运行           | 构建时额外消耗         |
| **Conda+部分Docker**   | 100%本地资源 | 应用在Conda，基础设施在Docker | 资源利用更灵活         |

跳过 jemalloc，Windows 不需要

运行命令删除整个 `.venv`文件夹：

Remove-Item -Recurse -Force .venv

*# 检查当前项目目录结构*

ls -Force

uv pip list | grep huggingface*# 检查 pyproject.toml 中的依赖是否都安装了*

ls -la download_deps.py

#### 解决huggingface_hub问题：

uv run python -c "import huggingface_hub; print('huggingface_hub 导入成功')"

uv run python -c "import huggingface_hub; print(huggingface_hub.__file__)"

PYTHONPATH=. uv run python download_deps.py

#### wsl走代理问题

ip route | grep default

export http_proxy=http://172.31.96.1:7890
export https_proxy=http://172.31.96.1:7890

root@mark-bunee:/mnt/d/ASUS/xiaozhi/ragflow-0.22.0# export http_proxy=http://172.31.96.1:7890
root@mark-bunee:/mnt/d/ASUS/xiaozhi/ragflow-0.22.0# export https_proxy=http://172.31.96.1:7890
root@mark-bunee:/mnt/d/ASUS/xiaozhi/ragflow-0.22.0# export HTTPS_PROXY=http://172.31.96.1:7890

curl https://google.com

*# 查看是否有 .venv 文件夹* 

ls .venv -Force

*# 查看 uv 使用的 Python 解释器路径* 

uv run python -c "import sys; print(sys.prefix)"

```
Python 项目管理生态
├── 环境管理
│   ├── Conda (重量级) ← 推荐用于复杂项目
│   └── venv (轻量级)
│
├── 包安装工具  
│   ├── uv (新一代，极速) ← 未来趋势
│   ├── pip (官方标准)
│   └── conda (conda生态专用)
│
└── 应用分发
    └── pipx (全局工具安装)
    
```

```
# 功能：安装Python包
pip install requests

# 特点：
# - Python官方标准工具
# - 只能管理Python包
# - 不处理环境隔离（需要配合venv）
# - 速度相对较慢

# 功能：安装可执行的Python应用
pipx install jupyter
pipx install black

# 特点：
# - 专门用于安装命令行工具
# - 自动创建隔离环境
# - 避免污染系统Python
# - 只能用于安装"应用"，不能用于开发

# 功能：快速安装和管理依赖
uv pip install requests
uv add requests  # 类似npm/yarn

# 特点：
# - 用Rust编写，速度极快
# - 整合了pip + venv的功能
# - 支持依赖解析缓存
# - 新兴工具，生态在完善中

# 功能：管理完整环境（包括系统库）
conda create -n myenv python=3.10
conda install numpy

# 特点：
# - 管理Python包 + 系统依赖（如C库）
# - 强大的环境隔离
# - 特别适合科学计算
# - 包数量比PyPI少但质量高
```

| 工具  | 主要用途     | 环境隔离   | 速度 | 适用场景           |
| :---- | :----------- | :--------- | :--- | :----------------- |
| pip   | 安装Python包 | ❌ 需要venv | 慢   | 基础Python开发     |
| pipx  | 安装全局工具 | ✅ 自动隔离 | 中等 | 命令行工具         |
| uv    | 快速依赖管理 | ✅ 内置     | 极快 | 现代Python项目     |
| conda | 完整环境管理 | ✅ 强大隔离 | 中等 | 数据科学、复杂依赖 |

使用windows 下载会出现

**`datrie`包在 Windows 上编译失败**是主要原因。这是一个经典的 Windows Python 包编译问题。

- `datrie`是一个包含 C 扩展的包，需要编译
- Windows 上需要 Microsoft Visual C++ Build Tools
- 错误信息中缺少关键的编译错误，但根本原因是缺少编译环境

```
sudo tee /etc/systemd/system/docker.service.d/proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=http://172.31.96.1:7890"
Environment="HTTPS_PROXY=http://172.31.96.1:7890"
Environment="NO_PROXY=localhost,127.0.0.1,.docker.io,.docker.com,.hub.docker.com"
EOF
```

sudo systemctl daemon-reload 

sudo systemctl restart docker

docker pull hello-world

sudo docker info | grep -A 10 "Registry Mirrors"

| 容器名称           | CONTAINER ID | CPU 使用率 | 内存使用/限制         | 内存占比 | 网络 I/O        | 块 I/O  | PIDS |
| :----------------- | :----------- | :--------- | :-------------------- | :------- | :-------------- | :------ | :--- |
| **docker-mysql-1** | 09654319a040 | 0.56%      | 197.3 MiB / 7.608 GiB | 2.53%    | 3.17MB / 2.36MB | 0B / 0B | 55   |
| **docker-es01-1**  | 83d7bfa69ce2 | 0.30%      | 3.285 GiB / 7.519 GiB | 43.69%   | 1.79MB / 23.7kB | 0B / 0B | 144  |
| **docker-minio-1** | e921ac139594 | 0.00%      | 257.7 MiB / 7.608 GiB | 3.31%    | 19.6MB / 4.19MB | 0B / 0B | 21   |
| **docker-redis-1** | d567f2644041 | 0.13%      | 7.668 MiB / 7.608 GiB | 0.10%    | 863kB / 660kB   | 0B / 0B | 5    |

| 组件类别       | 组件名称             | 内存占用    | 占总内存比例 | CPU使用率 | 状态评估       |
| :------------- | :------------------- | :---------- | :----------- | :-------- | :------------- |
| **主程序**     | ragflow_server.py    | 1.21 GB     | 15.9%        | 6.9%      | 🟡 正常         |
| **主程序**     | task_executor.py     | 1.23 GB     | 16.2%        | 未知      | 🟡 正常         |
| **主程序小计** | **RAGFlow主程序**    | **2.44 GB** | **32.1%**    | **-**     | **🟡 正常**     |
| **中间件**     | Elasticsearch (es01) | 3.285 GB    | 43.2%        | 0.30%     | 🔴 内存大户     |
| **中间件**     | MySQL                | 197.3 MB    | 2.5%         | 0.56%     | 🟢 正常         |
| **中间件**     | MinIO                | 257.7 MB    | 3.3%         | 0.00%     | 🟢 正常         |
| **中间件**     | Redis                | 7.67 MB     | 0.1%         | 0.13%     | 🟢 正常         |
| **中间件小计** | **所有中间件**       | **3.75 GB** | **49.3%**    | **-**     | **🔴 内存紧张** |
| **系统总计**   | **全部组件**         | **6.19 GB** | **81.4%**    | **-**     | **🔴 严重不足** |



cd /mnt/d/ASUS/ragflow-0.22.0/ragflow-0.22.0/docker

docker compose -f docker-compose-base.yml down

markbunee@mark-bunee:/mnt/d/ASUS/ragflow-0.22.0/ragflow-0.22.0/web$ npm run dev

docker compose -f docker/docker-compose-base.yml up -d

pkill -f "ragflow_server.py|task_executor.py"

## HR_resources_market

```
uvicorn sever:app --host 0.0.0.0 --port 8000 --reload

curl -X POST "http://localhost:8101/rank/biz" \
  -H "Content-Type: application/json" \
  -d '{
    "post_text": "门店招聘收银员与前厅服务员，需沟通能力与基础收银技能",
  }'
  
  curl -X POST "http://localhost:8101/rank/users" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "有三年餐饮服务经验，收银熟练",
  }'
```

## ragflow二次开发

### 启动

cd /mnt/d/ASUS/xiaozhi/ragflow-0.22.0

export PYTHONPATH=$(pwd)

export UV_INDEX=https://mirrors.aliyun.com/pypi/simple

source .venv/bin/activate

vim /etc/hosts

```
127.0.0.1       localhost
127.0.1.1       mark-bunee.     mark-bunee
127.0.0.1       es01 infinity mysql minio redis sandbox-executor-manager
```

bash docker/launch_backend_service.sh

python external/api/app.py

#### 测试

```

1）文档摘要功能
curl -X POST "http://localhost:8009/v1/abstract_extract/summary/extract" \
  -H "Authorization: Bearer ragflow-6i9ewRJz3x8y0Ggo-ZQMnED48KWBhXCYowLyY4Ah-KE" \
  --data "doc_id=299d5881dcaa11f0a7b0177fe4f11677"
  

2）获取doc_id
curl --request GET   --url "http://127.0.0.1:9380/api/v1/datasets?page=1&page_size=30"   -H "Authorization: Bearer ragflow-6i9ewRJz3x8y0Ggo-ZQMnED48KWBhXCYowLyY4Ah-KE"
// 获取dataset_id后获取doc_id
curl --request GET   --url "http://127.0.0.1:9380/api/v1/datasets/2393a034dcaa11f0a7b0177fe4f11677/documents?page=1&page_size=10"   -H "Authorization: Bearer ragflow-6i9ewRJz3x8y0Ggo-ZQMnED48KWBhXCYowLyY4Ah-KE"

3)json测试
python /mnt/d/ASUS/xiaozhi/ragflow-0.22.0/mxx_tools/pic/jsonread.py

4）知识库-搜索
curl -s -X POST "http://localhost:8009/v1/file_search/2393a034dcaa11f0a7b0177fe4f11677/retrieval" \
  -H "Authorization: Bearer ragflow-6i9ewRJz3x8y0Ggo-ZQMnED48KWBhXCYowLyY4Ah-KE" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "请帮我检索与 DeepSeek 相关的回答要点",
    "page": 1,
    "page_size": 10,
    "top_k": 50,
    "similarity_threshold": 0.2,
    "vector_similarity_weight": 0.3,
    "highlight": true
  }'
```

#### 问题与知识



























































