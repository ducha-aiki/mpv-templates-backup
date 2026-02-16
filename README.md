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

You can create conda environment with all required packages via the following for CPU:

```bash
conda create --name mpv-assignments python=3.13
conda activate mpv-assignments
# CPU-only wheels (adjust if a different wheel tag is required for your platform)
pip3 install torch==2.8.0 torchvision==0.23.0
pip3 install kornia==0.8.2 tqdm notebook matplotlib opencv-contrib-python==4.13.0.92 seaborn tensorboard tensorboardX ipywidgets widgetsnbextension
pip3 install kornia_moons --no-deps
```

And following for GPU. You may need to change the cuda version to the actually installed one.
For the GPU setup, if you have CUDA-capable GPU (if needed - change CUDA version in command). To find out your CUDA version, run `nvidia-smi`. You will see something like:

```
Mon Feb 20 16:49:46 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.39       Driver Version: 460.39       CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  On   | 00000000:01:00.0 Off |                  N/A |
| 25%   44C    P8    18W / 250W |      1MiB / 11178MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  On   | 00000000:06:00.0 Off |                  N/A |
|ERR!   54C    P0   ERR! / 250W |      1MiB / 11178MiB |     75%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
In the example above, the CUDA version is 11.2, so you should use `–extra-index-url https://download.pytorch.org/whl/cu128`



```bash
conda create --name mpv-assignments-gpu python=3.13
conda activate mpv-assignments-gpu
# Change the CUDA wheel tag to match your installed CUDA version (e.g. cu128)
pip3 install torch==2.8.0 torchvision==0.23.0 --extra-index-url https://download.pytorch.org/whl/cu128
pip3 install kornia==0.8.2 tqdm notebook matplotlib opencv-contrib-python==4.13.0.92 seaborn tensorboard tensorboardX ipywidgets widgetsnbextension
pip3 install kornia_moons --no-deps
```

For Apple Silicon devices (M1 and so on, family) use:

```bash
conda create --name mpv-assignments python=3.13
conda activate mpv-assignments
pip3 install torch==2.8.0 torchvision==0.23.0
pip3 install kornia==0.8.2 tqdm notebook matplotlib opencv-contrib-python==4.13.0.92 seaborn tensorboard tensorboardX ipywidgets widgetsnbextension
pip3 install kornia_moons --no-deps
```

**Keep in mind that the assignments and the assignment templates will be updated during the semester.  Always pull the current template version before starting to work on an assignment!**
