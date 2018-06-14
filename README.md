# Accurate-Binary-Convolution-Network  
Binary Convolution Network for faster real-time processing in ASICs  

---

Tensorflow implementation of [Towards Accurate Binary Convolutional Neural Network](https://arxiv.org/abs/1711.11294) by Xiaofan Lin, Cong Zhao, and Wei Pan.  
Why this network? Let's quote the authors
> It has been known that using binary weights and activations drastically reduce memory size and accesses, and can replace arithmetic operations with more efficient bitwise operations, leading to much faster test-time inference and lower power consumption.  
> The implementation of the resulting binary CNN, denoted as ABC-Net, is shown to achieve much closer performance to its full-precision counterpart, and even reach the comparable prediction accuracy on ImageNet and forest trail datasets, given adequate binary weight bases and activations.

### Dependencies
```sh
pip install -r requirements.txt
```
By default `tensorflow-gpu` will be installed. Make sure to have `CUDA` properly setup.

### Notebooks
* **ABC** - Contains the original implementation of the ABC network
* **ABC-layer-inference-support** - Slightly modified functions for better inference time support (tldr; moved the alpha training operation out of the layer)

### Testing
* MNIST - Accuracy on validation set reached upto 94%. (Check the notebook for information)
* ImageNet - To be added

### TODO
- [ ] Test on ImageNet (2012)
- [ ] Add visualization of the complete `ABC` layer