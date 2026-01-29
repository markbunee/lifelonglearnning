```
cd /mnt/c/Users/admin/Desktop/skime


wsl安装uv
# 使用 ghproxy 加速下载
curl -LsSf https://mirror.ghproxy.com/https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-musl.tar.gz | tar -xz -C /tmp

# 将下载好的文件移动到本地 bin 目录
mkdir -p ~/.local/bin
cp /tmp/uv-x86_64-unknown-linux-musl/uv* ~/.local/bin/

# 刷新环境
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc


wsl安装miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

uv pip install uvicorn fastapi

uvicorn main:app --host 0.0.0.0 --port 8000
npm run dev -- --host

npm install

uv run run.py --transport sse --port 8881
cd mcp_tool
cd repository_understand_mcp
```

