# 1.RAGFlow 二开小智知识库

## 1.1  部署：

部分网络清华源不可用情况可能出现

从您提供的路径可以看出，huggingface_hub 已经安装在虚拟环境的 site-packages 中，但是当运行 `uv run download_deps.py`时，却提示找不到模块。

可能的原因有：

1. 虚拟环境可能没有正确激活或 uv run 没有使用正确的虚拟环境。
2. 路径问题：可能当前工作目录不在项目根目录，或者 Python 路径设置有问题。
3. 模块名称大小写问题：在 Linux 系统中，模块名是大小写敏感的。您安装的包是 `huggingface_hub`，但导入时使用的是 `huggingface_hub`，这应该是一致的，但注意文件名是 `__init__.py`，所以模块名应该是 `huggingface_hub`。

huggingface_hub确实安装在虚拟环境中，但 Python 却找不到它。这通常是由于 Python 路径问题或模块损坏导致的。

检查模块是否真的存在

ls -la /mnt/d/ASUS/ragflow-0.22.0/ragflow-0.22.0/.venv/lib/python3.10/site-packages/huggingface_hub/

cat /mnt/d/ASUS/ragflow-0.22.0/ragflow-0.22.0/.venv/lib/python3.10/site-packages/huggingface_hub/__init__.py

测试直接导入

```
uv run python -c "
import sys
print('Python 路径:', sys.prefix)
print('sys.path:')
for p in sys.path:
    print('  ', p)

print('尝试导入...')
try:
    import huggingface_hub
    print('✓ 直接导入成功')
except ImportError as e:
    print('✗ 直接导入失败:', e)
"

which python
uv run which python
uv run python -c "import sys; print(sys.version); print(sys.executable)"

echo $PYTHONPATH
```

```
markbunee@mark-bunee:/mnt/d/ASUS/ragflow-0.22.0/ragflow-0.22.0$ uv run python -c "import sys, os; print('当前目录:', os.getcwd()); print('在 sys.path 中:', os.getcwd() in sys.path)"
Uninstalled 1 package in 75ms
░░░░░░░░░░░░░░░░░░░░ [0/1] Installing wheels...                                   warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
         If the cache and target directories are on different filesystems, hardlinking may not be supported.
         If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
Installed 1 package in 2.02s
当前目录: /mnt/d/ASUS/ragflow-0.22.0/ragflow-0.22.0
在 sys.path 中: False

解决方案：
PYTHONPATH=. uv run python download_deps.py
```

## 1.2 源码启动

```
cd /mnt/d/ASUS/xiaozhi/ragflow-0.22.0

export PYTHONPATH=$(pwd)

export UV_INDEX=https://mirrors.aliyun.com/pypi/simple

source .venv/bin/activate

export DOC_ENGINE=infinity  # 将es换成infinity 

bash docker/launch_backend_service.sh

python external/api/app.py

curl -X POST -F "file=@/mnt/d/ASUS/111.pdf" http://localhost:8009/v1/bioclip/upload

/mnt/d/ASUS/biorag/ragflow-0.22.0/external/api/apps/test_pdf_upload.py
```



```
vim /etc/hosts
127.0.0.1       localhost
127.0.1.1       mark-bunee.     mark-bunee
127.0.0.1       es01 infinity mysql minio redis sandbox-executor-manager
```

 export DOC_ENGINE=infinity   app.py读取setting文件

## 1.3 测试

```
1）文档摘要功能
curl -X POST "http://localhost:8009/v1/abstract_extract/summary/extract" \
  -H "Authorization: Bearer ragflow-6i9ewRJz3x8y0Ggo-ZQMnED48KWBhXCYowLyY4Ah-KE" \
  --data "doc_id=299d5881dcaa11f0a7b0177fe4f11677"
  

2）获取doc_id

curl --request GET   --url "http://127.0.0.1:9380/api/v1/datasets?page=1&page_size=30"   -H "Authorization: Bearer ragflow-6i9ewRJz3x8y0Ggo-ZQMnED48KWBhXCYowLyY4Ah-KE"
// 获取dataset_id后获取doc_id
curl --request GET   --url "http://127.0.0.1:9380/api/v1/datasets/5245826cde4911f0ad7485f5c7e80b5c/documents?page=1&page_size=10"   -H "Authorization: Bearer ragflow-6i9ewRJz3x8y0Ggo-ZQMnED48KWBhXCYowLyY4Ah-KE"

3)json测试
python /mnt/d/ASUS/xiaozhi/ragflow-0.22.0/mxx_tools/pic/jsonread.py

4）知识库-搜索
curl -s -X POST "http://192.168.30.214:8009/v1/file_search/retrieval" \
  -H "Authorization: Bearer ragflow-6i9ewRJz3x8y0Ggo-ZQMnED48KWBhXCYowLyY4Ah-KE" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "请帮我检索与 DeepSeek 相关的回答要点",
    "dataset_ids": ["4e8e1200dfad11f0bf97bfbc04264fe9"],
    "page": 1,
    "page_size": 10,
    "top_k": 50,
    "similarity_threshold": 0.2,
    "vector_similarity_weight": 0.3,
    "highlight": true
  }'
  
# ["4e8e1200dfad11f0bf97bfbc04264fe9", "5245826cde4911f0ad7485f5c7e80b5c"]

#  "doc_ids": ["<DOC_ID_1>", "<DOC_ID_2>"],

---------------
旧接口示例：
curl -s -X POST "http://localhost:8009/v1/file_search/retrieval"   -H "Authorization: Bearer ragflow-<API_KEY>"   -H "Content-Type: application/json"   -d '{
    "question": ""，   ##问题
    "page": 1,            ##页数功能: 结果分页的页码（从 1 开始）作用: 控制返回第几页的 chunks（每页 page_size 条）
    "dataset_ids": ["4e8e1200dfad11f0bf97bfbc04264fe9", "5245826cde4911f0ad7485f5c7e80b5c"], ##数据库传入
    "page_size": 10,#- 功能: 单页返回的结果条数- 作用: 直接影响响应体 chunks 的数量与延迟

    "top_k": 50, #初筛候选的最大条数（向量与文本融合前的候选上限）
    "similarity_threshold": 0.2, #相似度过滤阈值，低于该值的结果会被过滤，不计入 total ，也不进入 chunks
    "vector_similarity_weight": 0.3, #融合权重（向量相似度相对于关键词相似度的占比），越大越偏向语义匹配；越小越偏向关键词精确匹配
    "highlight": true #是否在返回中包含高亮文本片段，

    
  }'

-------
12月24日 搜索请求：
curl -s -X POST "http://localhost:8009/v1/file_search/retrieval" \
  -H "Authorization: Bearer ragflow-6i9ewRJz3x8y0Ggo-ZQMnED48KWBhXCYowLyY4Ah-KE" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "请帮我检索与 DeepSeek 相关的回答要点",
    "dataset_ids": ["4e8e1200dfad11f0bf97bfbc04264fe9", "5245826cde4911f0ad7485f5c7e80b5c"],
    "page": 1,
    "page_size": 5,
    "top_k": 10,
    "similarity_threshold": 0.2,
    "vector_similarity_weight": 0.3,
    "highlight": true,
    "summarize": true,
    "llm_id": "Qwen3-32B@OpenAI-API-Compatible",
    "temperature": 0.2,
    "top_p": 0.9,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "related": true,
    "mindmap": true,
    "meta_data_filter": {
      "method": "auto"
    }
  }'
```

## 1.4 docker日志查询

```
最新的 100 行并开始持续追踪，而不是从头看起，请使用 --tail 参数：
docker logs -f --tail 100 docker-ragflow-cpu-1
查看最近 30 分钟的日志：
docker logs --since 30m docker-ragflow-cpu-1
查看指定日期之后的日志：

```



## 1.5 问题与知识

# 2.VitaRAG

## 2.1 启动

```
cd /mnt/d/ASUS/biorag/ragflow-0.22.0

cd /mnt/d/ASUS/xiaozhi/ragflow-0.22.0

export PYTHONPATH=$(pwd)

export UV_INDEX=https://mirrors.aliyun.com/pypi/simple

source .venv/bin/activate

cd web
npm run dev
```

## 2.2 问题与知识

Q：pip install到项目文件夹的环境.env文件夹可以直接把代码和环境一起打包迁移到其他地方吗

A：不能，虽然 `.env` 文件夹（通常是 Python 的虚拟环境）里包含了运行代码的所有依赖库，但它具有**“位置依赖性”**。

**硬编码路径（Hardcoded Paths）：** 虚拟环境在创建时，会将 Python 解释器的绝对路径（例如 `D:\ASUS\xiaozhi\.env\bin\python`）写入环境内部的脚本中（如 `pip`、`activate` 脚本）。如果你迁移到另一台电脑或换了文件夹路径，这些脚本依然会去寻找旧的路径，导致环境报错。

**二进制兼容性：** 如果你是从 Windows 迁移到 Linux，或者从 Intel 芯片的电脑迁移到 M1/M2 芯片的 Mac，`.env` 里的二进制文件（`.exe` 或 `.so`）根本无法运行。

**系统库依赖：** 有些库（如 OpenCV、CUDA）依赖于系统安装的 C++ 运行库或驱动。只拷贝 `.env` 文件夹，这些系统级的支持并不会跟着走。

正确的迁移方案：重新构建

如果你迁移的环境没有网络，无法使用 `pip install`，有以下两种替代方案：

1.**Conda Pack (针对 Conda 环境)：** 如果你使用的是 Anaconda，可以使用 `conda-pack` 工具，它可以将环境打包成一个可移植的压缩包，并处理路径问题。

2.**Docker (推荐方案)：** 将你的代码、模型文件和环境全部打包成一个 **Docker 镜像**。这是目前解决“在我电脑上能跑，在你那里不行”的最佳方案。

## 2.3 启动

后端环境和前端环境和启动用ragflow的，加载源码进行

## 2.4 Biomedclip

```
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "D:/huggingface_cache"
```



# 3.脑机eeg图像文本生成统一多模态框架

```
模型缓存地址：
缓存目录（默认是 ~/.cache/huggingface/hub/）
临时处理配置源：
export http_proxy="http://127.0.0.1:7890" 

export https_proxy="http://127.0.0.1:7890"

export HF_ENDPOINT="https://hf-mirror.com"

pip install -r requirements_web_demo.txt -i https://mirrors.aliyun.com/pypi/simple/

source ~/.bashrc
模型烂尾后需要删除然后再下载：

rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B

pip install timm -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements_web_demo.txt -i https://mirrors.aliyun.com/pypi/simple/

5090堡垒机：
http://10.21.43.6/luna/?login_to=2b349f63-c434-46c3-b1b3-99cf456c4c0a
/home/temp_user/
~/MXX/eegtxtimgaes/Qwen2.5-Omni-main/low-VRAM-mode

conda activate eegbrain

python stepa_test.py

```

## 3.1 huggingface下载机制

Hugging Face 的下载机制分为三层：

**Blobs 文件夹**：存储真实的大文件数据。

**Snapshots 文件夹**：存储指向 Blobs 的“快捷方式”（软链接）。

**验证机制**：当你下载中断时，Blobs 里可能留下了一个大小不正确的文件。再次启动时，`transformers` 库检查发现 `model-00003...` 这个文件在索引里有，但本地实体文件是不完整的或不存在，就会抛出 `OSError`。

```
huggingface删除模型缓存：
cd ~/.cache/huggingface/hub
rm -rf models--Qwen--Qwen2.5-Omni-7B
```



## 3.2 学校远程机1080ti出现conda环境源出问题

```
conda create -n eegbrain python=3.10 --offline 使用缓存创建镜像源

权限与拒绝： 403 不同于 404（找不到），它是服务器明确告诉你“我知道你要什么，但我拒绝给你”。

Conda 环境的本质：
不仅仅是文件夹： 一个 Conda 环境必须包含 Python 解释器二进制文件（以及关联的库文件）。
隔离性： 当你 activate 环境时，系统只是修改了环境变量 $PATH，把新环境的 bin 文件夹放到了最前面。
为什么不能在空环境 pip： 如果环境里没有 Python，系统找不到当前环境的 pip，就会调用 base 或系统自带的 pip，导致包装错地方。
```

## 3.3 http状态码

#### 3.3.1 状态码分类（五大家族）

| **开头数字** | **类别**                      | **含义**                                    |
| ------------ | ----------------------------- | ------------------------------------------- |
| **2xx**      | 成功 (Success)                | 请求已成功处理（如 `200 OK`）               |
| **3xx**      | 重定向 (Redirection)          | 资源搬家了，需要去新地址找                  |
| **4xx**      | **客户端错误** (Client Error) | **你的问题**（地址填错、没权限、频率太快）  |
| **5xx**      | **服务器错误** (Server Error) | **网站的问题**（服务器宕机、代码写 Bug 了） |

#### 3.3.2 常见错误代码深度展开

1. 4xx 客户端错误（你需要调整操作）

- **400 Bad Request（语义错误）**
  - **现象：** 服务器不理解你的请求。
  - **知识点：** 通常是请求参数格式错误，或者是某些特殊字符没转义。在 Conda 中，如果配置文件格式写错了，可能会触发类似错误。
- **401 Unauthorized（未授权）**
  - **现象：** 需要登录才能访问。
  - **知识点：** 与 403 不同，401 意味着你**证明了身份**就能进。比如访问公司私有的 Conda 镜像源，需要配置用户名和密码。
- **404 Not Found（找不到资源）**
  - **现象：** 你请求的 URL 不存在。
  - **知识点：** 路径写错了。就像你之前尝试 `.../anaconda/pkgs/main` 报 404，是因为阿里云的真实路径可能多了一层或少了一层文件夹。
- **408 Request Timeout（请求超时）**
  - **现象：** 你的网太慢了，服务器等得不耐烦了。
  - **知识点：** 网络波动或跨境连接时常见。
- **429 Too Many Requests（请求过多）**
  - **现象：** 你被“封号/封 IP”了。
  - **知识点：** **反爬虫机制**。如果你短时间内频繁刷新镜像站，或者同一个局域网（比如实验室）有几十个人同时在跑 `conda install`，镜像站会暂时拉黑你们。

2. 5xx 服务器错误（你只能等或者换源）

- **500 Internal Server Error（内部错误）**
  - **现象：** 服务器崩了。
  - **知识点：** 对方服务器的程序出错了。
- **502 Bad Gateway（网关错误）**
  - **现象：** 服务器之间的“接力棒”掉了。
  - **知识点：** 镜像站通常有一个前端代理服务器（如 Nginx）和一个后端存储。502 说明前端能通，但后端存储挂了。**清华源维护时经常出现这个。**
- **503 Service Unavailable（服务不可用）**
  - **现象：** 服务器现在太忙（超载）或者正在停机维护。
- **504 Gateway Timeout（网关超时）**
  - **现象：** 前端代理等后端响应等到花儿都谢了。通常也是服务器负载过高导致的。

## 3.4 什么是 `repodata.json`？

当你运行 `conda install` 或 `conda create` 时，Conda 做的第一件事不是下载包，而是先去源地址下载这个名为 `repodata.json` 的文件。

这个文件里记录了该频道（Channel）下**所有包**的信息，包括：

- **包名**（如 `python`, `numpy`）

- **版本号**（如 `3.10.12`）

- **依赖关系**（如 `numpy` 需要某个版本的 `mkl`）

- **哈希值**（用于校验下载的文件是否完整）

  

## 3.5 Pip 临时换源：

```
使用 -i (全称 --index-url) 参数。

指令格式： pip install [包名] -i [镜像源地址]

常用国内源：

阿里： https://mirrors.aliyun.com/pypi/simple/

清华： https://pypi.tuna.tsinghua.edu.cn/simple

豆瓣： https://pypi.douban.com/simple/

进阶用法（信任非 HTTPS 源）： 如果源地址是 http 而非 https，可能需要加上信任参数： pip install numpy -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

导出与分享： pip freeze > requirements.txt
```

## 3.6 Conda 临时换源：

```
使用 -c (全称 --channel) 参数。

指令格式： conda install [包名] -c [镜像源地址] --override-channels

关键点： 必须带上 --override-channels，否则 Conda 还是会去默认源里转一圈，可能再次触发 403。

例子： conda install pytorch -c http://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/ --override-channels
```

### 3.6.1 Conda 进阶常用指令清单

#### 环境管理

| `conda env list`                        | 查看所有环境及存放路径                 |
| --------------------------------------- | -------------------------------------- |
| `conda create -n 新环境 --clone 旧环境` | **离线**复制环境（极其有用，不用联网） |
| `conda remove -n 环境名 --all`          | 彻底删除某个环境                       |
| `conda env export > env.yml`            | 导出当前环境的配置文件（方便分享）     |

#### 包与缓存清理

| **指令**           | **作用**                                 |
| ------------------ | ---------------------------------------- |
| `conda list`       | 查看当前环境下安装的所有包               |
| `conda clean -i`   | 清除索引缓存（解决找包慢、源报错的问题） |
| `conda clean -p`   | 删除没用的包（节省硬盘空间）             |
| `conda clean -all` | 彻底清理（当 Conda 抽风时的终极手段）    |

#### 配置查询

```
查看当前生效的所有配置： conda config --show-sources （这个能帮你找到到底是哪个文件锁定了清华源）

查看当前优先级： conda config --get channels
```



# 4.语音分离的数据库更新

bz_opinion_file @smart_app_bzopinion (smart_app_bzopinionyuqing) -表

bz_opinion_voice_analysis @smart app_bzopinion (smart app_bzopinionyuqing) - 表

/file-resource/1919643053772152833/bz_opinion_analysis_file/analysis/20250910234303-18180999219-S20250910234301470900AC130E1705370884-0000107800187006_1757665608957.mp3

```
查询特定位置的context是否被更改
SELECT 
    f.id, 
    f.file_url, 
    v.file_id, 
    v.context,
    -- 检查 JSON 是否合法：1 为合法，0 为非法
    JSON_VALID(v.context) AS is_json_valid,
    -- 计算解析出的对话条数
    JSON_LENGTH(v.context) AS dialog_count
FROM bz_opinion_file_copy1 f
LEFT JOIN bz_opinion_voice_analysis_copy1 v ON f.id = v.file_id
WHERE f.id = 1416097737165766656;
```

## 4.1 公司地址

cd /data/upload_file/05_zhishiku/mxx/updatevociecontext 公司地址192.168.30.214

## 4.2 配置miniconda开发环境

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 下载miniconda

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh 清华源

**安装**： `bash Miniconda3-latest-Linux-x86_64.sh`

**激活**： `source ~/miniconda3/bin/activate`

删除下载损坏的安装包 rm -f Miniconda3-latest-Linux-x86_64.sh 

删除安装失败的残留文件夹 rm -rf /root/miniconda3

source /root/miniconda3/bin/activate 如果source ~/.bashrc没反应手动激活

## 4.3 screen使用教程

```
问题：
(base) root@ituu:/data/upload_file/05_zhishiku/mxx# screen -r updatecontext
There is a screen on:
        316054.updatecontext    (12/22/2025 02:44:39 PM)        (Attached)
There is no screen to be resumed matching updatecontext.

解决方案：
强制剥离并连接（最推荐）
screen -d -r updatecontext
```

```
完整返回
{"status":"success","result":{"file":"20250811142333-18679590406-S20250811142333431037AC130E1711279928-0000107800184263_1763022114938.mp3","language":"zh","outputs":{"csv":"/app/results/ef1d0f4e-a9db-4db8-9de2-8d760c09b9e5_20250811142333-18679590406-S20250811142333431037AC130E1711279928-0000107800184263_1763022114938_with_roles.csv"},"output_contents":{"csv":[{"time":{"start":"00:00:00,060","end":"00:00:02,935"},"role":"游客","text":""},{"time":{"start":"00:00:00,180","end":"00:00:07,060"},"role":"客服","text":"你好,赛里木湖景区你好,我想问一下那个北门早上最早几点钟能进啊?"},{"time":{"start":"00:00:14,400","end":"00:00:19,820"},"role":"游客","text":"喂,你好哎,你好八点八点吗?"},{"time":{"start":"00:00:19,980","end":"00:00:23,360"},"role":"客服","text":"最早八点才能进嗯这是八点钟人工上班是吗?"},{"time":{"start":"00:00:23,400","end":"00:00:30,500"},"role":"客服","text":"嗯哦,那我们拿那个温泉县的那个发票是不是可以免一下门票然后这架票再另外付就可以了"},{"time":{"start":"00:00:31,660","end":"00:00:35,860"},"role":"游客","text":"就是那个,哦对,一张180一个门票"},{"time":{"start":"00:00:37,040","end":"00:00:41,500"},"role":"客服","text":"好的,就是一张180发票能免一个人对吧如果两张就是免两个人"},{"time":{"start":"00:00:42,180","end":"00:00:42,720"},"role":"游客","text":"嗯,是的"},{"time":{"start":"00:00:43,920","end":"00:00:45,140"},"role":"客服","text":"嗯,谢谢"}]}}}


```

## 4.4 docker部署

```
实时查看容器日志：
docker logs -f voice-sync-task
进入容器查看：
docker exec -it voice-sync-task /bin/bash
查看容器内的日志文件：cat logs/sync_task.log

查看环境变量是否传进来：env | grep DB

退出容器：输入 exit 或按 Ctrl + D

停止并清除容器：docker stop voice-sync-app 
docker rm voice-sync-app

docker restart

```

## 4.5 构建镜像

```
docker build -t voice-sync-task:v1 .
```

## 4.6 启动容器

```
docker run -d \
  --name voice-app_update_v2 \
  -e DB_HOST="192.168.30.62" \
  -e DB_PASS="smart_app_bzopinion@la22" \
  -e TABLE_FILE="bz_opinion_file" \
  -e TABLE_ANALYSIS="bz_opinion_voice_analysis" \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/updatecontent.py:/app/updatecontent.py \
  --restart always \
  voice-sync-task:v1
  
  
 生产： 
 cd /home/szbz/mxx_linshi/updatevociecontext/

  docker run -d \
  --name voice-app_update \
  -e DB_HOST="10.253.97.190" \
  -e DB_PASS="smart_app_bzopinion@la22" \
  -e TABLE_FILE="bz_opinion_file" \
  -e TABLE_ANALYSIS="bz_opinion_voice_analysis" \
  -e MINIO_URL="http://10.253.63.201:50079" \
  -e API_URL="http://10.253.63.200:50086/diarize" \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/updatecontent.py:/app/updatecontent.py \
  --restart always \
  voice-sync-task:v1
  
  支持挂载宿主机代码

```

## 4.7 查看运行日志

```
docker logs -f voice-sync-app
docker logs -f 3f1e569a3b2fa835a43d806d407064b492eb9b1cc4ed187d3a6e87b7634be89a
```



## 4.8 进入容器内部查看

```
docker exec -it voice-sync-app /bin/bash
```

## 4.9 生产docker部署（无网络环境）

```
docker build -t voice-sync-task:v1 .

docker save -o voice-sync-task_v1.tar voice-sync-task:v1

docker load -i voice-sync-task_v1.tar

docker images

docker run -d voice-sync-task:v1
```

## 4.10 传参样例

```
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "192.168.30.62"),   // 数据库位置
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "smart_app_bzopinion"),
    "password": os.getenv("DB_PASS", "smart_app_bzopinion@la22"),
    "database": os.getenv("DB_NAME", "smart_app_bzopinion"),
    "charset": "utf8mb4"
}
MINIO_BASE_URL = os.getenv("MINIO_URL", "http://192.168.30.165")  // 音频放置位置
API_URL = os.getenv("API_URL", "http://192.168.30.220:8000/diarize") // 语音分离服务请求



TABLE_FILE = os.getenv("TABLE_FILE", "bz_opinion_file_copy1")  地址的表
TABLE_ANALYSIS = os.getenv("TABLE_ANALYSIS", "bz_opinion_voice_analysis_copy1") context的表
```

bz_opinion_voice_analysis_copy1:

![image-20251223094058069](./pic/image-20251223094058069.png)

bz_opinion_file_copy1:

![](./pic/image-20251223094129944.png)

## 4.11 数据库查询：

```
SELECT 
    f.id, 
    f.file_url, 
    v.file_id, 
    v.context,
    -- 检查 JSON 是否合法：1 为合法，0 为非法
    JSON_VALID(v.context) AS is_json_valid,
    -- 计算解析出的对话条数
    JSON_LENGTH(v.context) AS dialog_count
FROM bz_opinion_file f
LEFT JOIN bz_opinion_voice_analysis v ON f.id = v.file_id
WHERE f.id = 1438564778296475648;
```





# 5.人声分离部署

```
cd /data/xyx/diarization-api
export HF_ENDPOINT="https://hf-mirror.com"
设置huggingface
export HF_HOME="/path/to/your/big_disk/huggingface"

开放环境启动容器：
docker run -d --name diarization_container \
  --gpus all \
  --network host \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -e CUDNN_LIB_DIR=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib \
  -e LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib \
  -v $(pwd)/results:/app/results \
  -v /data/xyx/diarization-api/huggingface:/root/.cache/huggingface \
  -v /data/xyx/diarization-api/torch:/root/.cache/torch \
  diarization-api:latest
  
  
测试：
  curl -X POST "http://localhost:8000/diarize"   -F "url=http://192.168.30.165/file-resource/1919643053772152833/bz_opinion_analysis_file/analysis/20250811142333-18679590406-S20250811142333431037AC130E1711279928-0000107800184263_1763022114938.mp3"   -F "language=zh"   -F "whisper_model=medium"   -F "device=cuda"   -F "no_stem=true" 
  
删除容器：
docker stop 497b1966f4c8
docker rm 497b1966f4c8

```

## 5.1 堡垒机权限设置与指令

```
写入权限被限制
sudo chmod 777 /data
```

## 5.2 生产部署

```
curl -X POST "http://10.253.63.200:50086/diarize"   -F "url=http://10.253.63.201:50079/file-resource/1919643053772152833/bz_opinion_analysis_file/analysis/20251213120942-15542659021-S20251213120942e62e96a93f79416d9-0000101004691001_1765599358973.mp3"   -F "language=zh"   -F "whisper_model=medium"   -F "device=cuda"   -F "no_stem=true"
```

## 5.3 Bash配置































