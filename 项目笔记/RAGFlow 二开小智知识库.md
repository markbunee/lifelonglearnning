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

