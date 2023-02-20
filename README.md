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

You can create conda environment with all required packages via  the following for CPU:

```bash
conda create --name mpv-assignments-cpu-only python=3.10
conda activate mpv-assignments-cpu-only
pip3 install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install kornia==0.6.10 tqdm notebook matplotlib opencv-contrib-python==4.7.0.68 seaborn tensorboard tensorboardX ipywidgets widgetsnbextension
pip3 install kornia_moons --no-deps
```

And following for GPU. You may need to change the cuda version to the actually installed one.

```bash
conda create --name mpv-assignments-gpu python=3.10
conda activate mpv-assignments-gpu
pip3 install torch==1.12.1+cu113 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install kornia==0.6.10 tqdm notebook matplotlib opencv-contrib-python==4.7.0.68 seaborn tensorboard tensorboardX ipywidgets widgetsnbextension
pip3 install kornia_moons --no-deps
```

For Apple Silicon devices (M1, M2 family) use:

```bash
conda create --name mpv-assignments-cpu-only python=3.10
conda activate mpv-assignments-cpu-only
conda install -c apple tensorflow-deps
pip3 install tensorflow-macos tensorflow-metal
pip3 install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install kornia==0.6.10 tqdm notebook matplotlib opencv-contrib-python==4.7.0.68 seaborn tensorboard tensorboardX ipywidgets widgetsnbextension
pip3 install kornia_moons --no-deps
```

**Keep in mind that the assignments and the assignment templates will be updated during the semester.  Always pull the current template version before starting to work on an assignment!**
