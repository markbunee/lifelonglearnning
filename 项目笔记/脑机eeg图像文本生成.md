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

python stepa_test.py  # 生成
python test_eegencoder.py --model_path /MXX/eegtxtimgaes/Qwen2.5-Omni-main/best_12.8_change.pth  #读取模型然后才是eeg编码器

#### 读取原始数据集
# 使用默认路径
python visualorigin_data.py

# 指定自定义路径
python visualorigin_data.py \
    --eeg_signals_path data/EEG_data/eeg_5_95_std.pth \
    --splits_path data/EEG_data/block_splits_by_image_single.pth \
    --output_dir outputs

# 只分析特定被试者
python code/visualize_dataset.py --subject 4


微调eeg编码器
cd MXX/eegtxtimgaes/Qwen2.5-Omni-main
conda activate eegbrain
python main.py
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

## Dreamdiffusion复现和检查：

```
python stageA1_eeg_pretrain.py \
  --root_path ~/DreamDiffusion-main \
  --roi VC \
  --num_epoch 20 \
  --batch_size 64 \
  --mask_ratio 0.1 \
  --patch_size 4 \
  --embed_dim 1024 \
  --decoder_embed_dim 512 \
  --depth 24 \
  --num_heads 16 \
  --decoder_num_heads 16 \
  --mlp_ratio 1.0 \
  --include_hcp True \
  --include_kam True

  
 # dreamdiffusion stageb
 python /home/temp_user/MXX/DreamDiffusion-main/code/eeg_ldm.py \
  --root_path /home/temp_user/MXX/DreamDiffusion-main \
  --pretrain_gm_path /home/temp_user/MXX/DreamDiffusion-main/pretrains \
  --pretrain_mbm_path "/home/temp_user/MXX/DreamDiffusion-main/checkpoint.pth" \
  --batch_size 4 \
  --lr 5.3e-5 \
  --precision 16 \
  --accumulate_grad 8 \
  --num_epoch 50 \
  --ddim_steps 250
  
  
 # 推理
 nohup python3 code/gen_eval_eeg.py --dataset EEG --model_path  "/home/temp_user/MXX/DreamDiffusion-main/checkpoint.pth" --splits_path "datasets/block_splits_by_image_single.pth" --eeg_signals_path "datasets/eeg_5_95_std.pth" --config_patch "pretrains/models/config15.yaml" --imagenet_path "datasets/imageNet_images" > eval.log 2>&1 &


  
```

