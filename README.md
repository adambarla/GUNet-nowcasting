# GUNet-nowcasting
My bachelor's thesis that focused on applying Guided Upsampling to UNet to improve weather nowcasting.

## Abstract
Short-term weather forecasting (nowcasting) is crucial in predicting extreme weather events. This thesis focuses on structural biases that can be introduced into convolutional neural network predictions.
I investigated whether guided upsampling, used instead of transposed convolution, can improve the accuracy and reliability of weather forecasts. To this end, I trained UNet and Guided UNet (GUNet) models on radar images from a network of meteorological radars created by the OPERA radar pro- gram. I analyzed both models using performance indicators such as mean squared error (MSE), mean absolute error (MAE), and the structural similar- ity index (SSIM).
The results showed that the GUNet model slightly outperformed the UNet model regarding average MAE and MSE and demonstrated a better ability to capture higher frequencies in the Fourier spectrum of radar images. More- over, the GUNet model achieved marginally better results on images with higher radar echo intensity, essential for predicting severe weather events.
The study suggests that the GUNet model can improve short-term weather predictions, and the results provide a basis for further research in this area.

__Keywords:__ weather nowcasting, convolutional neural networks, transposed convolution, UNet, guided UNet, guided filter, guided upsampling

## Contents
```
├─ src/...................................directory containing the implementation
│  ├─ dataset.py...............................script for generating the datasets
│  ├─ tuning.py..................................script for hyperparameter tuning
│  ├─ train.py............................................model trainining script
│  ├─ gunet.py.................................implementation of GUNet in PyTorch
│  ├─ unet.py...................................implementation of UNet in PyTorch
│  ├─ gunet_weights_best.pt..............................eights of the best GUNet
│  ├─ unet_weights_best..................................weights of the best UNet
│  ├─ visualization.ipynb...........notebook used for creating the visualizations
├─ thesis/..........................................directory containing the text
│  ├─ chapters directory..........................containing text of the chapters
│  ├─ images directory.......................containing images used in the thesis
│  ├─ thesis.tex...........................................main LaTeX source file
│  ├─ bibliography.bib...............................................bibliography
│  ├─ thesis.pdf........................................text of the thesis in pdf
├─ README.md........................information about the contents of the archive
```
