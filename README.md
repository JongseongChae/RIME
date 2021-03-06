# Robust Imitation learning with Multiple perturbed Environments (RIME)
This repository is an implementation of "Robust Imitation Learning against Variations in Environment Dynamics" accepted to ICML 2022.

The RIME code are modified from the codes of [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).


## Supported Environments
+ [MuJoCo](https://www.roboti.us/index.html) (via [OpenAI Gym](https://www.gymlibrary.ml/))

I only tested MuJoCo 200 and used MuJoCo tasks with perturbed dynamics. I made these MuJoCo tasks with perturbed dynamics by changing components of xml files for tasks I want to introduce perturbations.


## Requirements
I provide all libraries and packages for this codes. Try the follow
```
pip install -r requirements.txt
```


## Run Example 
For training and evaluating agents (via 10 random seeds). You can change env-parameter, algo-name as follows for selecting other dynamics perturbation type or testing other algorithms.
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
