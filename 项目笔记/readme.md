# 1.RAGFlow 二开小智知识库

## 部署：

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

## 源码启动

```
cd /mnt/d/ASUS/xiaozhi/ragflow-0.22.0

export PYTHONPATH=$(pwd)

export UV_INDEX=https://mirrors.aliyun.com/pypi/simple

source .venv/bin/activate

export DOC_ENGINE=infinity  # 将es换成infinity 

bash docker/launch_backend_service.sh

python external/api/app.py
```



```
vim /etc/hosts
127.0.0.1       localhost
127.0.1.1       mark-bunee.     mark-bunee
127.0.0.1       es01 infinity mysql minio redis sandbox-executor-manager
```

 export DOC_ENGINE=infinity   app.py读取setting文件

## 测试

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

## 问题与知识

# 2.VitaRAG

## 启动

```
cd /mnt/d/ASUS/biorag/ragflow-0.22.0

cd /mnt/d/ASUS/xiaozhi/ragflow-0.22.0

export PYTHONPATH=$(pwd)

export UV_INDEX=https://mirrors.aliyun.com/pypi/simple

source .venv/bin/activate

cd web
npm run dev
```

## 问题与知识

Q：pip install到项目文件夹的环境.env文件夹可以直接把代码和环境一起打包迁移到其他地方吗

A：不能，虽然 `.env` 文件夹（通常是 Python 的虚拟环境）里包含了运行代码的所有依赖库，但它具有**“位置依赖性”**。

**硬编码路径（Hardcoded Paths）：** 虚拟环境在创建时，会将 Python 解释器的绝对路径（例如 `D:\ASUS\xiaozhi\.env\bin\python`）写入环境内部的脚本中（如 `pip`、`activate` 脚本）。如果你迁移到另一台电脑或换了文件夹路径，这些脚本依然会去寻找旧的路径，导致环境报错。

**二进制兼容性：** 如果你是从 Windows 迁移到 Linux，或者从 Intel 芯片的电脑迁移到 M1/M2 芯片的 Mac，`.env` 里的二进制文件（`.exe` 或 `.so`）根本无法运行。

**系统库依赖：** 有些库（如 OpenCV、CUDA）依赖于系统安装的 C++ 运行库或驱动。只拷贝 `.env` 文件夹，这些系统级的支持并不会跟着走。

正确的迁移方案：重新构建

如果你迁移的环境没有网络，无法使用 `pip install`，有以下两种替代方案：

1.**Conda Pack (针对 Conda 环境)：** 如果你使用的是 Anaconda，可以使用 `conda-pack` 工具，它可以将环境打包成一个可移植的压缩包，并处理路径问题。

2.**Docker (推荐方案)：** 将你的代码、模型文件和环境全部打包成一个 **Docker 镜像**。这是目前解决“在我电脑上能跑，在你那里不行”的最佳方案。

## 启动

后端环境和前端环境和启动用ragflow的，加载源码进行

# 3.脑机eeg图像文本生成统一多模态框架

临时处理配置源：

export http_proxy="http://127.0.0.1:7890" 

export https_proxy="http://127.0.0.1:7890"

pip install -r requirements_web_demo.txt -i https://mirrors.aliyun.com/pypi/simple/

source ~/.bashrc

```
modeling_qwen2_5_omni_low_VRAM_mode.py

Qwen2RMSNorm 的功能是均方根归一化 (Root Mean Square Normalization) 。简单来说，它就像是神经网络中的“稳定器”或“调节阀”。
- 稳定数值 ：在深度神经网络（特别是像 Qwen 这样的大模型）中，数据经过很多层计算后，数值可能会变得非常大或非常小（梯度爆炸或消失）。RMSNorm 把这些数值拉回到一个合理的范围内。
- 加速训练 ：通过规范化数据的分布，让模型更容易学习，收敛得更快。
它与传统 LayerNorm 的区别（为什么叫 RMS）：

- 传统 LayerNorm ：先减去平均值（Center），再除以标准差（Scale）。
- RMSNorm ： 不减去平均值 ，直接除以均方根（RMS）。
  - 好处 ：计算量更少，速度更快，而且实验证明在 Transformer 模型中效果一样好甚至更好。
比喻： 想象一个班级的考试成绩，有的考 100 分，有的考 10 分，差异很大。

- LayerNorm 像是先把全班平均分平移到 0 分，再缩放。
- RMSNorm 则是直接把大家的分数按比例缩放，让分数的“能量”（平方和）保持在一个标准水平，不管你原来均值是多少。
```



|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |
|      |      |      |
|      |      |      |

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

## 公司地址

cd /data/upload_file/05_zhishiku/mxx/updatevociecontext 公司地址192.168.30.214

## 配置miniconda开发环境

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 下载miniconda

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh 清华源

**安装**： `bash Miniconda3-latest-Linux-x86_64.sh`

**激活**： `source ~/miniconda3/bin/activate`

删除下载损坏的安装包 rm -f Miniconda3-latest-Linux-x86_64.sh 

删除安装失败的残留文件夹 rm -rf /root/miniconda3

source /root/miniconda3/bin/activate 如果source ~/.bashrc没反应手动激活

## screen使用教程

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

## docker部署

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

## 构建镜像

```
docker build -t voice-sync-task:v1 .
```

## 启动容器

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

## 查看运行日志

```
docker logs -f voice-sync-app
docker logs -f 3f1e569a3b2fa835a43d806d407064b492eb9b1cc4ed187d3a6e87b7634be89a
```



## 进入容器内部查看

```
docker exec -it voice-sync-app /bin/bash
```

## 生产docker部署（无网络环境）

```
docker build -t voice-sync-task:v1 .

docker save -o voice-sync-task_v1.tar voice-sync-task:v1

docker load -i voice-sync-task_v1.tar

docker images

docker run -d voice-sync-task:v1
```

## 传参样例

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

## 数据库查询：

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

## 堡垒机权限设置与指令

```
写入权限被限制
sudo chmod 777 /data
```

## 生产部署

```
curl -X POST "http://10.253.63.200:50086/diarize"   -F "url=http://10.253.63.201:50079/file-resource/1919643053772152833/bz_opinion_analysis_file/analysis/20251213120942-15542659021-S20251213120942e62e96a93f79416d9-0000101004691001_1765599358973.mp3"   -F "language=zh"   -F "whisper_model=medium"   -F "device=cuda"   -F "no_stem=true"
```



































