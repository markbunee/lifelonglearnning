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

