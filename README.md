# SARLA

## Cross Modal UAV Object Tracking

![cross_modal](pic/cross_modal.gif)


## Install the environment
**Option1**: Use the Anaconda (CUDA 10.2)
```
conda create -n ostrack python=3.8
conda activate ostrack
bash install.sh
```

**Option2**: Use the Anaconda (CUDA 11.3)
```
conda env create -f ostrack_cuda113_env.yaml
```

**Option3**: Use the docker file

We provide the full docker file here.


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Download the dataset from [Baidu Drive](https://pan.baidu.com/s/1WwXkKUAwLHj3_lx1oKekdw?pwd=pykw ) 


## Training
Download pre-trained [OSTrack](https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd?usp=sharing)  and put it under `$PROJECT_ROOT$/pretrained_models`.

```
./train.sh
```


## Evaluation
Download the model weights from [Baidu Drive](https://pan.baidu.com/s/1WwXkKUAwLHj3_lx1oKekdw?pwd=pykw ) 

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/sarla`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

```
./test.sh
```


## Acknowledgments
* Thanks for the [STARK](https://github.com/researchmm/Stark) ,  [PyTracking](https://github.com/visionml/pytracking) and [OSTrack](https://github.com/botaoye/OSTrack) library, which helps us to quickly implement our ideas.
* We use the implementation of the ViT from the [Timm](https://github.com/rwightman/pytorch-image-models) repo.  
