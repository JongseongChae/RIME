# Robust Imitation learning with Multiple perturbed Environments (RIME)
This repository is an implementation of "Robust Imitation Learning against Variations in Environment Dynamics" accepted to ICML 2022.

The RIME code are modified from the codes of [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).


## Supported Environments
+ [MuJoCo](https://www.roboti.us/index.html) (via [OpenAI Gym](https://www.gymlibrary.ml/))

For perturbed tasks, I only used `mujoco200` and made these MuJoCo tasks with perturbed dynamics by changing components in xml files for the tasks to introduce fixed dynamics perturbations. For more details, please go to the `environments` folder.


## Requirements
I provide all libraries and packages for this codes.
```
pip install -r requirements.txt
```


## Run Example 
For training agents (over 10 random seeds), we can change `env-parameter`, `algo-name` for selecting other dynamics perturbation type (for single dynamics parameter cases) or training other algorithms as follows:
+ env-parameter: gravity / mass
+ algo-name: RIME / RIME+WSD / OMME / GAIL-Mixture / GAIL-Single
```
# train the agent in the 2 sampled interaction environments setting
python main.py --env-name=Hopper-v2 --env-parameter=gravity --sampled-envs=2 --algo-name=RIME+WSD

# train the agent in the 3 sampled interaction environments setting
python main.py --env-name=Hopper-v2 --env-parameter=gravity --sampled-envs=3 --algo-name=RIME+WSD

# train the agent in the 4 sampled interaction environments setting (2-dim perturbation parameter case)
python main.py --env-name=Hopper-v2 --sampled-envs=4 --algo-name=RIME+WSD
```
