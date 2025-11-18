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

















































