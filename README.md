# Squeezenext_baseline_Pytorch
Squeezenext baseline architecture implementation in Pytorch with live loss and accuracy update using livelossplot package.

__Software used__ : <br />
Python 3.6 <br />
Pytorch 1.0.0 <br />
Spyder 3.3.1 <br />

__Packages needed__ when running it for first time : <br />
Livelossplot package (https://github.com/stared/livelossplot) <br />
Torch (https://pytorch.org/get-started/locally/) <br />
- Choose all the options according to your system specification.<br />
- __Recommendations__ <br />
OS: Linux <br />
BUILD : 1.0 <br />
Package : Pip <br />
Cuda Installation; it depends on your graphic card. Please refer your label<br />


|-----Model-------|Kernel size-|-epochs-|test accuracy||Model Size(MB)||Model speed||Optimizer||Learning rate| <br />
|sqnxt_baseline_23|     7x7    |  200 	|--84.69%-----|-----2.617-----|-----22-----|----sgd---|-----0.1------| <br />
|sqnxt_baseline_23|     3x3    |  200	  |--87.63%-----|-----2.586-----|-----23-----|----sgd---|-----0.1------| <br />
![alt text](https://github.com/Jayan-K-Duggal/Squeezenext_baseline_Pytorch/blob/master/fig_plot_sqnxt_baseline.jpg)<br />
