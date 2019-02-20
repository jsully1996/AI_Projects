

## Overview
This is the code for my group course project for the course CMPT 726(Machine Learning) at Simon Fraser University. We have used a locally optimized version of a conditional GAN to generate synthetic car logo images from the existing dataset. Complete details can be found in the [project report](https://github.com/jsully1996/AI_Projects/Synthetic_Image_Generation_with_AC-GANs/report.pdf).


## Dependencies  
```
Tensorflow (1.0+)
Numpy
Python 3+
```
## Usage
```
Resize images before feeding to generator
>>>python resize.py

Convert 4D image to 3D
>>>python reduce_dim.py

Execute file with Python 3+
>>>python logo_gen.py
```



## Credits

The code for the generator and discrimator is from [moxiegushi](https://github.com/moxiegushi/pokeGAN). 
The implementation is using a heavily optimised version of AC-GANs instead of the original WGAN.
Full details of the changes can be found in the report.pdf file