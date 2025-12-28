# Cross-Embodiment Active Visual Tracking via Context-Aware Adaptation

## Installation

Our model rely on the DEVA as the vision foundation model and Gym-Unrealcv as the evaluation environment, which requires to install three additional packages: Grounded-Segment-Anything, DEVA and Gym-Unrealcv. Note that we modified the original DEVA to adapt to our task, we provide the modified version in the repository.

**Prerequisite:**
- Python 3.9
- PyTorch 2.0.1+ and corresponding torchvision
- gym_unrealcv(https://github.com/zfw1226/gym-unrealcv)
- Grounded-Segment-Anything (https://github.com/hkchengrex/Grounded-Segment-Anything)
- DEVA (https://github.com/hkchengrex/Tracking-Anything-with-DEVA)

**Clone our repository:**
```bash
git clone https://github.com/carponter/X_CAT_Tracking.git
```
**Install Grounded-Segment-Anything:**  
```bash
cd Offline_RL_Active_Tracking
git clone https://github.com/hkchengrex/Grounded-Segment-Anything

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda/
# /path/to/cuda/ is the path to the cuda installation directory, e.g., /usr/local/cuda
# if you install the cuda in conda, it should be {path_to_conda}/env/{conda_env_name}/lib/, e.g., ~/anaconda3/env/offline_evt/lib/

cd Grounded-Segment-Anything
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install --upgrade diffusers[torch]
```
**Install DEVA:**  
Directly install the modified DEVA in the repository  
(If you encounter the `File "setup.py" not found` error, upgrade your pip with `pip install --upgrade pip`)
```bash
cd ../Tracking-Anything-with-DEVA # go to the DEVA directory
pip install -e .
bash scripts/download_models.sh #download the pretrained models
```

**Install Gym-Unrealcv:**
```bash
cd ..
git clone https://github.com/zfw1226/gym-unrealcv.git
cd gym-unrealcv
pip install -e .
```
Before running the environments, you need to prepare unreal binaries. You can load them from clouds by running load_env.py
```bash
python load_env.py -e {ENV_NAME}

# To run the demo evaluation script, you need to load the UrbanCityMulti environment and textures by running:
python load_env.py -e UrbanCityMulti
python load_env.py -e Textures
sudo chmod -R 777 ./   #solve the permission problem
```
**Install mbrl:**
```bash
pip install mbrl==0.1.5
pip install pyglet==2.1.11 gym==0.10.9
```
