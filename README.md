# Hologram Reconstruction

This repository contains code to perform computational reconstruction of digital holograms. It was started as a branch of [HoloPy](https://github.com/manoharan-lab/holopy/tree/master), which is developed by the Manoharan Lab at Harvard University. Therefore, there are significant similarities between this code and that of HoloPy.

The focus of the repository is on hologram reconstruction, although the code can also be used to generate holograms (albeit computationally inefficiently).

The easiest way to start using the code is to look at the examples in the `examples` folder:
-  `reconstruction.py` shows the basic process of loading a hologram, performing some operations such as background subtraction, doing computational reconstruction and saving the results.
- `generation.py` shows how to use the hologram generation functionality.

Most classes and functions are thoroughly documented so it should be easy to understand and build on top of it.

Some of the functionality (specially the computational techniques for twin image elimination) is not fully tested, which is why this is not yet packaged.