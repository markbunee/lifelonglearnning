# 命令集

## 1.资料

[(90 封私信 / 68 条消息) Windows + WSL2 安装 Docker 并配置国内镜像源教程 - 知乎](https://zhuanlan.zhihu.com/p/1946945462901933435)



## 2.笔记

### 2.1 Linux

### 2.1.1 WSL命令

在cmd开启的命令：

wsl --status

wsl --list --verbose

查看发行版

wsl --install -d Ubuntu-22.04

wsl --shutdown 关闭所有实例

wsl --set-default-version 2

wsl --list --all  

wsl --distribution == wsl -d <发行版名称>

wsl --unregister <发行版名称>

当出现问题：

C:\Users\ASUS>wsl 无法将磁盘“C:\Users\ASUS\AppData\Local\Docker\wsl\main\ext4.vhdx”附加到 WSL2： 系统找不到指定的文件。 错误代码: Wsl/Service/CreateInstance/MountVhd/HCS/ERROR_FILE_NOT_FOUND

wsl --set-default Ubuntu-22.04



WSL2配置docker：

ping www.baidu.com ping registry-1.docker.io 检查wsl是否能联网

```text
sudo apt update
sudo apt install -y docker.io
sudo service docker start
sudo mkdir -p /etc/docker
sudo nano /etc/docker/daemon.json
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
sudo service docker restart
docker info
Registry Mirrors:
 https://docker.1ms.run
 https://docker.xuanyuan.me
docker pull hello-world
```

cat /proc/sys/vm/max_map_count

sysctl vm.max_map_count

sudo sysctl -w vm.max_map_count=262144

echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.conf

sudo sysctl -p

在windows资源管理器地址栏输入\\\wsl$即可查看wls系统文件

`\\wsl$\` 是Windows访问WSL存储的"入口"，**实际文件在Windows的AppData中**

查看docker位置：docker info | grep "Docker Root Dir"

Windows子系统，不是完整虚拟机

**迁移WSL到D盘**

wsl –shutdown

复制整个Packages文件夹到D盘:

Copy-Item -Path "C:\Users\markbunee\AppData\Local\Packages\*" -Destination "D:\WSL"

创建符号链接（让Windows以为还在C盘）

mklink /J "C:\Users\markbunee\AppData\Local\Packages" "D:\WSL\Packages"

### conda

conda env list

conda activate

conda create -n mxx python=3.10

conda doctor

conda remove -n mxx –all

conda clean --all

conda config --show

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

conda config --set show_channel_urls yes

pip cache purge

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

pip config set global.trusted-host mirrors.aliyun.com

pip install -i https://mirrors.aliyun.com/pypi/simple/ 包名 --trusted-host mirrors.aliyun.com

对于指定库，三种github镜像：

pip install git+https://gitclone.com/github.com/MahmoudAshraf97/demucs.git 

pip install git+https://ghproxy.com/https://github.com/oliverguhr/deepmultilingualpunctuation.git

pip install git+https://kgithub.com/MahmoudAshraf97/ctc-forced-aligner.git

ubantu配置conda环境：

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh

source ~/.bashrc

conda --version

### 挂后台指令

nohup uvicorn server:app --host 0.0.0.0 --port 8123 > server.log 2>&1 &

screen/tmux方法：

conda install screen

screen -S diarization

uvicorn server:app --host 0.0.0.0 --port 8123

Ctrl + A, 然后按 D

[detached from 12345.diarization]

netstat -tulpn | grep 8123

curl -X POST "http://localhost:8123/diarize" \  -F "url=http://192.168.30.165/file-resource/1919643053772152833/bz_opinion_analysis_file/analysis/20250811142333-18679590406-S20250811142333431037AC130E1711279928-0000107800184263_1763022114938.mp3" \  -F "language=zh" \  -F "whisper_model=medium" \  -F "device=cuda" \  -F "no_stem=true"

screen -ls

screen -r diarization

### screen

| 类别               | 命令                          | 说明                                                         |
| :----------------- | :---------------------------- | :----------------------------------------------------------- |
| **创建与管理会话** | `screen -S <name>`            | **创建一个名为 `<name>`的新 screen 会话。**（最常用，推荐总是给会话起名） |
|                    | `screen -d -S <name>`         | 创建一个名为 `<name>`的新会话，但**不**立即进入，而是使其在后台运行（Detached）。 |
|                    | `screen -r <name>`            | **恢复/连接到**一个名为 `<name>`的已断开（Detached）的会话。 |
|                    | `screen -ls`或 `screen -list` | **列出**当前所有的 screen 会话及其状态（Attached 或 Detached）。 |
| **会话内操作**     | `Ctrl + a`然后 `d`            | **从当前会话中分离（Detach）**。会话会在后台继续运行。这是最关键的快捷键。 |
|                    | `Ctrl + a`然后 `c`            | 在当前会话内**创建一个新的窗口（Window）**。                 |
|                    | `Ctrl + a`然后 `n`            | 切换到下一个窗口（Next）。                                   |
|                    | `Ctrl + a`然后 `p`            | 切换到上一个窗口（Previous）。                               |
|                    | `Ctrl + a`然后 `"`            | 列出所有窗口，供你选择切换。                                 |
|                    | `Ctrl + a`然后 `A`            | 为当前窗口重命名（Title）。                                  |
|                    | `exit`或 `Ctrl + d`           | **关闭当前窗口**。如果这是会话的最后一个窗口，则会话将被终止（结束）。 |
| **高级管理**       | `screen -x <name>`            | **强行接入**一个已经有人连着的会话（Attached）。用于会话共享和监视。 |
|                    | `screen -r <pid>.<tty>`       | 当有多个同名会话时，使用 `screen -ls`显示的 PID 来指定连接。 |
|                    | `Ctrl + a`然后 `S`            | **水平分割**当前窗口（大写的 S）。                           |
|                    | `Ctrl + a`然后 `Tab`          | 在分割的区域之间切换焦点。                                   |
|                    | `Ctrl + a`然后 `X`            | **关闭**当前焦点所在的区域。                                 |



| 类别           | 快捷键                | 说明                                                      |
| :------------- | :-------------------- | :-------------------------------------------------------- |
| **基本操作**   | `nano <文件名>`       | 打开（或创建）一个文件进行编辑。                          |
|                | `^O`                  | **保存文件**（Write **O**ut）。系统会提示你确认文件名。   |
|                | `^X`                  | **退出** nano。如果文件已修改，会询问你是否保存。         |
|                | `^G`                  | 打开**帮助文档**，查看所有命令的完整列表。                |
| **光标移动**   | 方向键                | 上下左右移动光标。                                        |
|                | `^A`                  | 移动到当前行的**行首**。                                  |
|                | `^E`                  | 移动到当前行的**行尾**。                                  |
|                | `^Y`                  | 向上**翻页**（Page Up）。                                 |
|                | `^V`                  | 向下**翻页**（Page Down）。                               |
|                | `^_`                  | **跳转到指定行和列**。按 `Ctrl + _`，然后输入行号，回车。 |
| **编辑文本**   | `Backspace`/ `Delete` | 删除光标前/后的字符。                                     |
|                | `^K`                  | **剪切**（Kill）当前整行。也用于标记后剪切。              |
|                | `^U`                  | **粘贴** 刚才剪切的内容到光标位置。                       |
|                | `Alt + ^`             | 复制当前行（相当于剪切再立即粘贴回去）。                  |
| **搜索与替换** | `^W`                  | **搜索**（Where is）文本。                                |
|                | `^`                   | **搜索并替换**文本。                                      |
|                | `Alt + W`             | 搜索**下一个**匹配项（在执行搜索命令后使用）。            |
| **高级功能**   | `^C`                  | 显示当前光标所在的**行号和列号**。                        |
|                | `^T`                  | 打开拼写检查功能（如果系统支持）。                        |
|                | `^R`                  | 将另一个文件的**内容插入**到当前光标位置。                |
|                | `^+`/ `Alt + +`       | 增大字体（如果终端支持）。                                |
|                | `^-`/ `Alt + -`       | 减小字体（如果终端支持）。                                |

## wsl 迁移

wsl --shutdown

wsl -l -v

wsl --export Ubuntu D:\wsl_backup\ubuntu.tar（将整个发行版导出为一个 $\text{.tar}$ 文件到临时位置。）

**注销原发行版**：这将删除 $\text{C}$ 盘上的原始 $\text{VHDX}$ 文件，释放空间。**请确保您已完成导出！**

wsl --unregister Ubuntu

wsl --import Ubuntu D:\wsl\Ubuntu D:\wsl_backup\ubuntu.tar --version 2（将 $\text{.tar}$ 文件导入到 $\text{D}$ 盘的目标文件夹，这将创建新的 $\text{VHDX}$ 文件。）



























