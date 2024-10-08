# **Meta-DT: Offline Meta-RL as Conditional Sequence Modeling with World Model Disentanglement**
Zhi Wang, Li Zhang, Wenhao Wu, Yuanheng Zhu, Dongbin Zhao, Chunlin Chen
A link to our paper can be found on 
## **Overview**
![MetaDT](./Meta-DT.jpg)
## **Installation**
Experiments require MuJoCo. Follow the instructions in the https://github.com/openai/mujoco-py  to install.
Create the virtual environment using 
See `requirments.txt` file for more information about how to install the dependencies.
```python
conda create -n meta_dt python=3.8.18 -y
conda activate meta_dt
pip install -r requirements.txt
```
## **Downloads Datasets and pretrained world model**
 - We share our datasets via this [Google Drive link]()
 - We share our pretrained world model via this [Google Drive link]()
```
TODO
```
## **Run Experiments**
Train the context encoder using world model 
```
python train_context.py --env_name AntDir-v0
```

Train the Meta Decision Transformer for few_shot Meta-DT
```
python train_meta_dt.py --env_name AntDir-v0 --zero_shot False --data_quality medium 
```
Train the Meta Decision Transformer for zero_shot Meta-DT
```
python train_meta_dt.py --env_name AntDir-v0 --zero_shot True --data_quality medium
```
