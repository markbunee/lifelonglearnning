# 命令集

## 1.资料

## 2.笔记

### 2.1 Linux

### 2.1.1 WSL命令

wsl --status

wsl --list --verbose

查看发行版

wsl --install -d Ubuntu-22.04

wsl --shutdown 关闭所有实例

wsl --set-default-version 2

当出现问题：

C:\Users\ASUS>wsl 无法将磁盘“C:\Users\ASUS\AppData\Local\Docker\wsl\main\ext4.vhdx”附加到 WSL2： 系统找不到指定的文件。 错误代码: Wsl/Service/CreateInstance/MountVhd/HCS/ERROR_FILE_NOT_FOUND

wsl --set-default Ubuntu-22.04