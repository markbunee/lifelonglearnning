# 阴道微生态菌群计数分析平台

## 本地运行（Ubuntu）

- 安装系统依赖

  ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip
  ```

- 创建虚拟环境并安装依赖

  ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -U pip
   pip install "gradio>=4,<5" "opencv-python>=4.8,<5" "numpy>=1.24,<2" "requests>=2,<3"
  ```

- 配置后端接口（项目根目录创建/编辑 config.json）

  ```json
   {
     "api": {
       "gelan_url": "http://<后端>/gelan",
       "shipian_url": "http://<后端>/shipian"
     },
     "device": {
       "company_name": "金域医疗",
       "machine_id": "test_01",
       "machine_ip": "127.0.0.1",
       "tenant_id": "tenant_01"
     }
   }
  ```

- 启动前端

  ```bash
   python ./count.py
  ```

  默认访问 [http://127.0.0.1:7860](http://127.0.0.1:7860/)；若需外网/容器访问，请在代码中使用：

  ```python
   demo.launch(server_name="0.0.0.0", server_port=7860)
  ```

## Docker 部署（Ubuntu）

- 构建镜像

  ```bash
   docker build -t kingmed-gradio:latest -f Dockerfile .
  ```

- 运行容器（映射端口与配置）

  ```bash
   docker run --rm -p 7860:7860 \
     -v "$(pwd)/config.json:/app/config.json" \
     -v "$(pwd)/count.py:/app/count.py" \
     --name kingmed-gradio kingmed-gradio:latest
  ```

  访问 [http://localhost:7860](http://localhost:7860/)。若未能访问，请确认 count.py 的 server_name 已设置为 "0.0.0.0"。

  ```
   生产部署
   
   docker save -o kingmed-frontend.tar kingmed-gradio:latest
   
   docker load -i kingmed-frontend.tar
  ```

```
 sudo docker ps -a | grep frontend
 
 先docker stop 再docker rm
 
sudo docker run -d -p 9313:7860 \
  -v "$PWD:/app" \
  --name kingmed-gradio kingmed-gradio:latest

sudo docker exec -it kingmed-gradio /bin/bash

http://10.132.90.6:9313

sudo docker logs -f kingmed-gradio

ps aux | grep python

curl http://10.132.90.6:9313

ls -l /app
cat /app/config.json


sudo docker logs -f kingmed-gradio


sudo docker run -d -p 9313:7860   -e PYTHONUNBUFFERED=1   -e GRADIO_ANALYTICS_ENABLED=false   -v "$(pwd):/app"   --name kingmed-gradio kingmed-gradio:latest


 sudo docker run -it --name ana_mxx_container -v /data/data_home/kmcv/czy/ana_training/mt_mxx/multcdi_label:/prj -v /Pathology_Al_Data_test/KMCV/ANA_data/0-kangrun/0-原始图片库:/home/kmcv --shm-size 256G --gpus 'device=1' yolo_images /bin/bash

```



ANA

```
 .\.venv\Scripts\activate
```
