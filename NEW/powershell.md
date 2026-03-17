 PowerShell = 命令行 + 脚本语言 + 自动化框架  

## PowerShell 与 Linux Shell 的区别 ：
PowerShell **不是文本流，是对象流**

powershell的脚本文件： .\script.ps1  

 PowerShell 本质上是 .NET 的脚本接口。  来自 Microsoft 的Microsoft .NET 框架。  

| 项 | PowerShell | Bash |
| --- | --- | --- |
| 输出 | 对象 | 文本 |
| 语言 | .NET | Shell |
| 平台 | 跨平台 | Linux |
| 命令结构 | Verb-Noun | 短命令 |


 PowerShell 命令结构 ：

 PowerShell 的命令叫Cmdlet    Verb-Noun  

| 命令 | 含义 |
| --- | --- |
| Get-Process | 查看进程 |
| Get-Service | 查看服务 |
| Get-ChildItem | 查看文件 |
| Stop-Process | 停止进程 |


## Powershell常见命令
```python
查看文件
Get-ChildItem
查看进程
Get-Process
查看服务
Get-Service
下载文件
iwr https://example.com/file.zip
iwr https://example.com/file.zip -OutFile file.zip

```

## PowerShell 管道  
```python
pipeline
|
Get-Process | Where-Object {$_.CPU -gt 100}
获取进程 → CPU>100 的

与linux不一样的地方 powershell

```















































