# Squeezenext_baseline_Pytorch
Squeezenext baseline architecture implementation in Pytorch with live loss and accuracy update using livelossplot package.

Software used : <br />
Python 3.6 <br />
Pytorch 1.0.0 <br />
Spyder 3.3.1 <br />
Livelossplot package (https://github.com/stared/livelossplot) <br />
Cuda 8.1 INSTALLED <br />

|-----Model-------|Kernel size-|-epochs-|test accuracy||Model Size(MB)||Model speed||Optimizer||Learning rate| <br />
|sqnxt_baseline_23|     7x7    |  200 	|--84.69%-----|-----2.617-----|-----22-----|----sgd---|-----0.1------| <br />
|sqnxt_baseline_23|     3x3    |  200	  |--87.63%-----|-----2.586-----|-----23-----|----sgd---|-----0.1------| <br />
![alt text](https://github.com/Jayan-K-Duggal/Squeezenext_baseline_Pytorch/blob/master/fig_plot_sqnxt_baseline.jpg)<br />
