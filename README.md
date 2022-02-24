
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/">
    <img src="logo-image/methods.png" alt="Logo" width="120" height="120">
  </a>

  <h3 align="center">Data-centric ML in QIS</h3>

Author: [Sanjaya Lohani](https://sanjayalohani.com)

*Please report bugs at slohani@mlphys.com

Thanks to [Brian T. Kirby](https://briankirby.github.io/), [Ryan T. Glasser](http://www.tulane.edu/~rglasser97/), [Sean D. Huver](https://developer.nvidia.com/blog/author/shuver/) and [Thomas A. Searles](https://ece.uic.edu/profiles/searles-thomas/)

Preprint:

1.  Lohani, S., Lukens, J.M., Glasser, R.T., Searles, T.A. and Kirby, B.T., 2022. Data-Centric Machine Learning in Quantum Information Science. arXiv preprint arXiv:2201.09134.

## Packages

```tensorflow >=2.4
qiskit
mlphys```


<!-- GETTING STARTED -->
## Getting Started

```pip install mlphys```

<!-- USAGE EXAMPLES -->
## Usage

```sh
import mlphys.deepqis.Simulator.Distributions as dist
import mlphys.deepqis.Simulator.Measurements as meas
import mlphys.deepqis.utils.Alpha_Measure as find_alpha
import mlphys.deepqis.utils.Concurrence_Measure as find_con
import mlphys.deepqis.utils.Purity_Measure as find_pm
import mlphys.deepqis.network.Inference as inference
import mlphys.deepqis.utils.Fidelity_Measure as fm
...
```

## Tutorials
_For examples (google colab), please refer to_ 
* [Generating Biased Distributions](https://github.com/slohani-ai/machine-learning-for-physical-sciences/blob/main/mlphys/deepqis/Biased_distributions_random_Q_states.ipynb). 
* [Inference Examples](https://github.com/slohani-ai/machine-learning-for-physical-sciences/blob/main/mlphys/deepqis/Inference_examples.ipynb).

## Upcoming Features
Ideal Measurements, Measurements at the given shots -- NISQ, 
Entangled States, Engineered Random Quantum States, 
Maximum Likelihood Estimation, 
Measurements on NISQ devices, 
more Pre-trained Models, ...
<!--
_open in the google colab_
* [Generating Biased Distributions]
* [Inference_Examples]
-->
