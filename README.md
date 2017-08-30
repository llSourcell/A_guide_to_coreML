# A_guide_to_coreML
This is the code for "A Guide to CoreML for iOS" by Siraj Raval on Youtube

## Overview 

This is the code for [this](https://youtu.be/T4t73CXB7CU) video on Youtube by Siraj Raval. The SMS dataset is [here](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/). This is a guide to Core ML, the new ML framework from Apple. The notebook explains it and the Xcode project implements a spam classifier. 

## Dependencies 

Install these with [pip](https://pip.pypa.io/en/stable/)
* scikit-learn
* numpy
* coremltools 

These are Apple specific
* Xcode 9.0+
* iOS 11+ to build on device

## Usage

Run the [notebook](http://jupyter.readthedocs.io/en/latest/install.html) using `jupyter notebook` in terminal. You can run open the iOS code by double clicking on the .xcodeproj file from [Xcode](https://developer.apple.com/xcode/downloads/). 

## Credits

The credits for this code go to [Gokul](https://github.com/gkswamy98). I've merely created a wrapper to get people started. 
