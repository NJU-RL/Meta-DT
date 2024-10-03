<<<<<<< HEAD
# test
=======
# Meta-DT
Experiments require MuJoCo. Follow the instructions in the https://github.com/openai/mujoco-py  to install.
Create the virtual environment using `conda env create -f environment.yaml`.
conda activate your_env
pip install -r requirements.txt
## Data collection 




## Train the context encoder using world model 

run `python train_context.py --env_name AntDir-v0` to train a context encoder and a world model decoder

## The Meta Decision Transformer (DT) algorithm
(i) for few_shot Meta-DT
 run `python train_meta_dt.py --env_name AntDir-v0 --zero_shot False --data_quality medium` to train the meta-DT algorithm using the pretrained context encoder, the `return/test tasks` in the tensorboard is the few-shot/zero-shot performance on the 5 unseen tasks during testing.
(ii) for zero_shot Meta-DT
run `python train_meta_dt.py --env_name AntDir-v0 --zero_shot True --data_quality medium`







>>>>>>> f632c32 (init)
