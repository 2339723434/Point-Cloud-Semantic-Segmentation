# Point Cloud Semantic Segmentation Backbone
轻量级点云语义分割网络，网络框架基于PointNet++，增加了显式几何辅助、对称式特征采样、及自注意力加权的局部特征学习，在保持较低参数的同时能达到较高的分割水平，适用于大规模场景点云语义分割。



### Installation

```
conda create -n pcs python=3.9
conda activate pcs
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install tensorboardx
pip install sharedarray
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.0%2Bcu111.html

cd modules/pointops
python3 setup.py install

cd ../pointops
python3 setup.py install
```



### Dataset prepration

```
cd data/S3DIS
gdown https://drive.google.com/u/1/uc?id=1UDM-bjrtqoIR9FWoIRyqLUJGyKEs22fP
tar zxf s3dis.tar.gz && rm s3dis.tar.gz && cd -
```



### Train

- To train one model for S3DIS Area-5:

  ```
  sh scripts/s3dis/train_s3dis.sh
  ```

  
