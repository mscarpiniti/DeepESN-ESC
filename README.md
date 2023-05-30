# DeepESN-ESC
A deep version of Echo State Network (DeepESN) for Environmental Sound Classification (ESC) with specific application to construction sites.

A DeepESN consists in the cascade of several reservoirs whose hidden states are concatenated in a unique and high-representative state vector used as input to a consecutive classifier, selected among several ones (i.e., linear, kNN, MLP, SVM, and random forest).

The proposed model is based on the TensorFlow implementation of the Deep Reservoir Computing introduced by [Claudio Gallicchio](https://github.com/gallicch/DeepRC-TF) in:
[1] Gallicchio, C., Micheli, A., Pedrelli, L.: Deep reservoir computing: A critical experimental analysis. Neurocomputing 268, 87–99 (2017). https://doi.org/10.1016/j.neucom.2016.12.08924.
