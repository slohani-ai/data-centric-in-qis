
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/">
    <img src="logo-image/methods.png" alt="Logo" width="120" height="120">
  </a>

  <h3 align="center">Data-centric ML in Quantum Information Science</h3>

Author: [Sanjaya Lohani](https://sanjayalohani.com)

*Please report bugs at slohani@mlphys.com

Thanks to [Brian T. Kirby](https://briankirby.github.io/), [Ryan T. Glasser](http://www.tulane.edu/~rglasser97/), [Sean D. Huver](https://developer.nvidia.com/blog/author/shuver/) and [Thomas A. Searles](https://ece.uic.edu/profiles/searles-thomas/)

Preprint:

Lohani, S., Lukens, J.M., Glasser, R.T., Searles, T.A. and Kirby, B.T., 2022. Data-Centric Machine Learning in Quantum Information Science. arXiv preprint arXiv:2201.09134.

## Built With
* [mlphys](https://pypi.org/project/mlphys/)
* [tensorflow >=2.4](https://www.tensorflow.org/)
* [qiskit](https://qiskit.org)



<!-- GETTING STARTED -->
## Getting Started

```pip install mlphys```

<!-- USAGE EXAMPLES -->
## Usage
### Simulating various distributions and measurements (including inferences)
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
### Sub-module
* <a href="https://github.com/slohani-ai/machine-learning-for-physical-sciences/tree/main/mlphys/deepqis">deepQis</a>

### Tutorials
_For examples (google colab), please refer to_ 
* [Generating Biased Distributions](https://github.com/slohani-ai/machine-learning-for-physical-sciences/blob/main/mlphys/deepqis/Biased_distributions_random_Q_states.ipynb). 
* [Inference Examples](https://github.com/slohani-ai/machine-learning-for-physical-sciences/blob/main/mlphys/deepqis/Inference_examples.ipynb).

## Hands-on coding examples for the results
* **Reducing spurious correlations:**
    * Accuracy of entanglement-separability classification - [Fig 2 (a)](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Toy-model/MEMS/plots/Fig%202%20a.ipynb)
    * network reconstruction fidelity versus the percentage of separable states added to a training set containing entangled states - [Fig 2 (b)](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Toy-model/MEMS/plots/Fig%202%20b.ipynb)
    * Reconstruction fidelity for test states from the MA distribution for a MEMS-only trained network and after adding a small
fraction of separable states into the training set - [Fig 2 (c, d)](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Toy-model/CP_werner_with_MA/Fig%202%20c%20and%20d.ipynb)
* **Reconstruction fidelity versus number of trainable parameters for various training set distributions:**
    * Data-centric approach (Fidelity versus trainable parameters) - [Fig 3 (a)](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/plots/Fidelity_vs_trainable_params/Fig%203.ipynb)  
    * The concurrence and purity of random quantum states from the Hilbert–Schmidt–Haar (HS–Haar), Zyczkowski (Z), ˙
engineered, and IBM Q distributions - [Fig 3 (a) insets](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/plots/histograms/Fig%203%20insets.ipynb)
* **Engineered states on concurrence-purity plane:**
    * The engineered and IBM Q sets - [Fig 3 (b)](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Toy-model/CP_werner_with_MA/Fig%204.ipynb)
* **Data-centric approach in the low-shot regime:**
    * Reconstructing the NISQ-sampled distribution with simulated measurements performed with shots ranging from 128 to 8192 - [Fig 4](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/shots_vary/plots/Fig%205.ipynb)  
* **Heterogeneous state complexity:**
    * Two-qubits
        * Reconstruction fidelities versus test state purity - [Fig 5 (a) bottom](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_train_MA_test/two%20qubits/plots/Fig%206%20(a)%20bottom.ipynb) 
        * Test MA distribution - [Fig 5 (a) top](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_train_MA_test/two%20qubits/plots/Fig%206%20(a)%20Density%20plot.ipynb) 
        * Fidelity versus K parameter - [Fig 5 (a) inset](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_train_MA_test/two%20qubits/plots/Fig%206%20(a)%20fid%20vs%20K.ipynb)
        * Zoomed in at the crossing point - [Fig 5 (a) inset](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_train_MA_test/two%20qubits/plots/Fig%206%20zoomed%20in.ipynb)
        
     * Three-qubits
        * Reconstruction fidelities versus test state purity - [Fig 5 (b) bottom](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_train_MA_test/three%20qubits/plots/Fig%206%20b%20fidelity%20vs%20purity.ipynb) 
        * Test MA distribution - [Fig 5 (b) top](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_train_MA_test/three%20qubits/plots/Fig%206%20b%20Density%20plot.ipynb) 
        * Fidelity versus K parameter - [Fig 5 (b) inset](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_train_MA_test/three%20qubits/plots/Fig%206%20b%20fidelity%20vs%20K.ipynb)
        * Zoomed in at the crossing point - [Fig 5 (b) inset](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_train_MA_test/three%20qubits/plots/Fig%206%20b%20zoomed%20in.ipynb)
      
     * Four-qubits
        * Reconstruction fidelities versus test state purity - [Fig 5 (c) bottom](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_train_MA_test/four%20qubits/plots/Fig%206%20c%20fideliy%20vs%20purity.ipynb) 
        * Test MA distribution - [Fig 5 (c) top](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_train_MA_test/four%20qubits/plots/Fig%206%20c%20Density%20plot.ipynb) 
        * Fidelity versus K parameter - [Fig 5 (c) inset](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_train_MA_test/four%20qubits/plots/Fig%206%20c%20fideliy%20vs%20K.ipynb)
        * Zoomed in at the crossing point - [Fig 5 (c) inset](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_train_MA_test/four%20qubits/plots/Fig%206%20c%20zoomed%20in.ipynb)
* **Engineered states:**
    * Unfiltered - [Fig 7 (a) left](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/brute_force_distro_gen/plots/Fig%207%20a.ipynb)
    * Engineered - [Fig 7 (a) right](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/brute_force_distro_gen/plots/Fig%207%20a.ipynb)
    * Reconstruction fidelities versus the value of K - [Fig 7 (b)](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/brute_force_distro_gen/plots/fid_vs_K_con_vs_K_pur_K/Fig%207%20b.ipynb)

* **Optimizing learning rate:**
    * Fidelity of reconstructed density matrices versus learning rate - [Fig 8](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/learning_rate_test/plots/Fig%208%20fidelity%20vs%20learning%20rate.ipynb)
    * The full purity distributions of the reconstructed states - [Fig 8 inset](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/learning_rate_test/plots/Fig%208%20Density%20vs%20purity.ipynb) 
    
* **Reconstruction fidelity versus trainable parameters for various MA-distributed training sets:**
    * The pairs of concentration parameter and K-value are chosen as (α, K) ∈ {(0.01, 4),(0.1, 4),(0.3, 4),(0.8, 4),(0.3394, 6)} for
training sets - [Fig 9](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/plots/Fidelity_vs_trainable_params/Fig%209.ipynb)

* **Reconstruction fidelity of NISQ-sampled test set versus the mean purity of various MA-distributed training states when K = 4.**
    * The mean purity of the training set matches the minimum and mean purity of the NISQ sampled states when D = K = 4 - [Fig 10](https://github.com/slohani-ai/data-centric-in-qis/blob/master/Distributions/Simulation/MA_fid_vs_purity_K_6/plots/Fig%2010.ipynb)

