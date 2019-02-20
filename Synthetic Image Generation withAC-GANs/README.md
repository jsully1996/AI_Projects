# pokeGAN

## Overview
This is the code for [this](https://youtu.be/yz6dNf7X7SA) video on Youtube by Siraj Raval. We'll use a WGAN to create new kinds of Pokemon. 

## Dependencies  
```
Tensorflow (1.0+)
Numpy
Python 3+
```
## Usage
```
Resize images before feeding to generator
python resize.py
Convert 4D image to 3D
python reduce_dim.py
Execute file with Python 3+
python logo_gen.py
```



## Credits

The code for the generator and discrimator is from [moxiegushi](https://github.com/moxiegushi/pokeGAN). 
The implementation is using a heavily optimised version of AC-GANs instead of the original WGAN.
Full details of the changes can be found in the report.pdf file