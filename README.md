# MPV python assignment templates
## Using this repo
The recommended way of using the templates is as follows.

First, you clone the repository:
```bash
git clone 
```

Then, you create a new branch for your solutions:
```bash
cd mpv-python-assignment-templates
git checkout -b solutions
```

After that, you can work on your solutions, commiting as necessary.

In order to update the template, commit all your work and execute:
```bash
# download the new template version:
git checkout master
git pull
# and update your solutions branch:
git checkout solutions
git merge master
```

You can create conda environment with all required packages via 

```bash
cd conda_env_yaml
# For CPU-only setup run
conda env create -f environment-cpu.yml
# if you have CUDA GPU card use instead
conda env create -f environment-gpu.yml
```

If way above does not work for you (e.g. you are on Windows), try the following for CPU:

```bash
conda create --name mpv-assignments-cpu-only python=3.9
conda activate mpv-assignments-cpu-only
pip3 install torch==1.8.2+cpu torchvision==0.9.2+cpu -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip3 install kornia==0.6.3 tqdm notebook matplotlib opencv-contrib-python==4.5.3.56 seaborn tensorboard tensorboardX
pip3 install kornia_moons --no-deps
conda install -c conda-forge widgetsnbextension
conda install -c conda-forge ipywidgets
```

And following for GPU:

```bash
conda create --name mpv-assignments-gpu python=3.9
conda activate mpv-assignments-gpu
pip3 install torch==1.8.2+cu102 torchvision==0.9.2+cu102  -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip3 install kornia==0.6.3 tqdm notebook matplotlib opencv-contrib-python==4.5.3.56 seaborn tensorboard tensorboardX
pip3 install kornia_moons --no-deps
conda install -c conda-forge widgetsnbextension
conda install -c conda-forge ipywidgets
```

**Keep in mind that the assignments and the assignment templates will be updated during the semester.  Always pull the current template version before starting to work on an assignment!**
