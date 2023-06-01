# DeepESN-ESC
A deep version of Echo State Network (DeepESN) for Environmental Sound Classification (ESC) with specific application to construction sites, as presented in [1].

A DeepESN consists in the cascade of several reservoirs whose hidden states are concatenated in a unique and high-representative state vector used as input to a consecutive classifier, selected among several ones (i.e., linear, kNN, MLP, SVM, and random forest).

The proposed model is based on the TensorFlow implementation of the Deep Reservoir Computing introduced by [Claudio Gallicchio](https://github.com/gallicch/DeepRC-TF) in [2]

[1] Scarpiniti, M., Perticarà, S., Lee, Y.-C., Uncini, A.: Deep Echo State Network for Environmental Sound CLassification. Submitted to *31st Edition of Italian Workshop on Neural Networks (WIRN 2023)* (2023).

[2] Gallicchio, C., Micheli, A., Pedrelli, L.: Deep reservoir computing: A critical experimental analysis. *Neurocomputing*, Vol. 268, pp. 87–99 (2017). https://doi.org/10.1016/j.neucom.2016.12.08924.
