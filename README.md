# ncs-object-detection

Python script for object detection using Intel Movidius Neural Compute Stick and a pretrained model.
## Installation

I tested the code on VirtualBox with Raspberry Pi Desktop. I installed Version 1 of the NCS SDK according to this tutorial https://www.bouvet.no/bouvet-deler/adding-ai-to-edge-devices-with-the-movidius-neural-compute-stick. 
This is also an useful guide for getting started.

*  NCS Virtual Machine configuration steps can be found here https://movidius.github.io/ncsdk/vm_config.html

## Getting started

The code is based on the security-cam app from ncapzoo https://github.com/movidius/ncappzoo/tree/ncsdk1

Run main.py for person detection of the sample image included in the /images folder. Feel free to modify the script, for example change the model (remember to update also the labels file, dimensions and the color mode).

```python
python main.py
```

## Technologies

* Python 2.7
* Intel NCS SDK
* MobileNet SSD for object detection
* VirtualBox with Raspberry Pi Desktop
