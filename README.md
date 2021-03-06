# DeepMaterial
This repository contains the code I wrote during my summer abroad at Stanford Radiological Sciences Lab.
I conducted my bachelors thesis on dual-energy material decomposition using deep learning. 
This research effort was supported with a grant by BaCaTec.

The project was logically structured into data generation and model fitting. 
First, the simulation environment [CONRAD](https://www.github.com/maxrohleder/CONRAD) was used to generate realistic dual 
energy data and corresponding material domain labels.
Second, the code in this project was used to wrap the data as a torch dataset and then train a Unet. 
The results were presented in my bachelor's thesis.

For more detailed insight, visit my [website](https://www.maxrohleder.de).

## Code Overview
- [stanford_env.yml](./stanford_env.yml)
    - all python dependencies used in this work.
- [CONRADataset.py](./CONRADataset.py)
    - wrapping raw files into a pytorch dataset. folder structure is assumed to be in the style as generated by the 
    CONRAD implementation    
- [train.py](./train.py)
    - the training procedure (not the cleanest code, but feel free to adapt to your needs)
- [gridSearch.py](./gridSearch.py)
    - my work included a hyperparameter optimization for the used Unet-model.
- [labelHistogram.py](./labelHistogram.py)
    - used this to analyze the simulated data. It plots the occurring pathlengths of materials of the labels. This could
    explain the performance variance in the results. Only those test images, which featured objects with material thickness
    in ranges corresponding to the test set performed well. 
- [cluster](./cluster)
    - this folder contains shell scripts to start computation on the gpu-cluster running slurm
