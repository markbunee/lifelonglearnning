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




```



ANA

```
 sudo docker run -it --name ana_mxx512_container -v /data/data_home/kmcv/czy/ana_training/mt_mxx/multi_label:/prj -v /Pathology_Al_Data_test/KMCV/ANA_data/0-kangrun/0-原始图片库:/home/kmcv --shm-size 512G --gpus all yolo_images /bin/bash
 
 .\.venv\Scripts\activate
sudo docker exec -it ana_mxx_container /bin/bash
 
nohup python train_best_acc.py --csv_path ./raw_csv/阳性数据训练测试集划分_v2_260323_bs2-bs7_with_multi_hot_train_val.csv > log_0323_new_$(date +%Y%m%d_%H%M%S).log 2>&1 &

vit
python d:\ASUS\kingmednan\multi_label\train_best_acc.py ^
  --csv_path d:\ASUS\kingmednan\raw_csv\你的数据.csv ^ 
  --model vit_b16 ^
  --epochs 50 ^
  --batch_size 64 ^
  --lr 0.0003
  
nohup python train_dulmodel.py --csv_path ./raw_csv/阳性数据训练测试集划分_with_multi_hot_train_val.csv > log_rs50_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train_best_acc.py --csv_path ./raw_csv/阳性数据训练测试集划分_with_multi_hot_train_val.csv > log_img512rs50_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

```
python huafen.py \
  --dataset_dir /Pathology_Al_Data_test/KMCV/ANA_data/0-kangrun/0-dsDNA_图片库/0-POC/dsDNA/dsDNA \
  --output_dir /data/data_home/kmcv/czy/ana_training/mt_mxx/multi_label/cls/datasets_split \
  --neg_name 阴性 \
  --pos_name 阳性 \
  --train_ratio 0.8 \
  --val_ratio 0.2

# 阳是0，阴是1 0323改回来0是阴1是阳
# 0 c-ANCA 1 p-ANCA
python D:\MXX\kingmednan\cls\huafen.py --dataset_dir D:\MXX\kingmednan\cls\datasetabca\cls_p_n --output_dir D:\MXX\kingmednan\cls\datasetabca\dataset_p_n --neg_name 0 --pos_name 1 --train_ratio 0.7 --val_ratio 0.3


python resnet_train.py 

```

```
python huafen.py --dataset_dir /data/data_home/kmcv/czy/ana_training/mt_mxx/multi_label/cls/cls_c_p --output_dir /data/data_home/kmcv/czy/ana_training/mt_mxx/multi_label/cls/dataset_c_p --neg_name 1 --pos_name 0 --train_ratio 0.7 --val_ratio 0.3



导包 wsl
pip download \
  scikit-learn==1.3.2 \
  numpy==1.26.4 \
  scipy==1.11.4 \
  joblib==1.3.2 \
  threadpoolctl==3.2.0 \
  -d sklearn_pkg \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --python-version 310 \
  --implementation cp \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com



pip install --no-index --find-links=sklearn_pkg scikit-learn




pip download \
  imagecodecs \
  -d imagecodecs_pkg \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --python-version 310 \
  --implementation cp \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com
  
pip install --no-index --find-links=imagecodecs_pkg imagecodecs



pip download \
  segmentation-models-pytorch==0.3.3 \
  timm==0.9.2 \
  -d smp_pkg \
  --no-deps \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --python-version 310 \
  --implementation cp \
  --abi cp310 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/


pip download \
  efficientnet-pytorch==0.7.1 \
  pretrainedmodels==0.7.4 \
  safetensors \
  pyyaml \
  munch \
  -d smp_pkg \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/
  
  
  pip install --no-index --find-links=smp_pkg segmentation-models-pytorch
```



cpython、协程进程线程效率优化、Python后端、OpenVINO模型推理框架、docker部署运维、LangChain、RAG、向量数据库、多Agent、Skills、大模型微调流程， SFT



机器学习、深度学习、TensorFlow和PyTorch框架、NLP、SD、VAE、COX生存分析、生境分析、因果推断、数据分析、数据挖掘
