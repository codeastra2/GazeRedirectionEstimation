# Gaze Estimation for Full Face Images

This repository contains the implementation to develop a gaze estimator for 128x128 full face images. 
It also contains the error analysis performed on the predictions of the same. 

### Requirements

This code was developed and tested on Ubuntu 18.04.5 LTS, it requires GPUs and [CUDA support](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
The python version used was 3.8.0 .
To run the code you would need [Jupyter Software](https://jupyter.org/). You can also configure [custom kernels](https://srinivas1996kumar.medium.com/adding-custom-kernels-to-a-jupyter-notebook-in-visual-studio-53e4d595208c#75c0-35c7b2ec5d54) for it in VS code.

All other requirements can be installed with:
`pip install -r requirements.txt`

#### Datasets/ Pre Processed Files required
`GazeCapture.h5, MPIIGaze.h5` are the training and testing sets used respectively. To obtain these please follow steps mentioned [here](https://github.com/swook/faze_preprocess) to pre process the original datasets.

`Redirected_samples1.h5`, is the augmented dataset please use [this](https://github.com/zhengyuf/STED-gaze) to generate the augmented dataset. 

For doing the error analysis, the predictions are loaded from `loss_2303196.pkl` pickle file. Please use the code in `train_gaze_estimator.ipynb` to generate predictions on the augmented dataset. 
### 


### Training and Evaluation
The notebook `train_gaze_estimation.ipynb` contains the code to train/ test the model, the cells are organized so that one may perform both separately. 
The resultant plots are saved in `plots` folder. 

### Error Analysis
The notebook `error_analysis.ipynb` contains the code to perform error analysis on the predictions. 

### Pre Trained Model
* [Model trained on Augmented Dataset, for a total of 20k Images and 20 epochs](https://drive.google.com/file/d/1lg-UJVBRvowWhWWPXRjxvC9PbSnKRgog/view?usp=sharing)

### References

* [Paper](https://arxiv.org/abs/2010.12307)
* [Gaze Redirection Code Repository](https://github.com/zhengyuf/STED-gaze)
* [Gaze Capture Dataset](https://gazecapture.csail.mit.edu/download.php)
* [MPIIGaze Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild)





















