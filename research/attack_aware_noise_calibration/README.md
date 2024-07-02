# Paper Experiments

### Installing Dependencies for Reproducing Experiments in the Paper

```
poetry install --with experiments --with dev
```

### Reproducing Plots in the Paper

Run:
```
poetry run jupytext --to ipynb research/attack_aware_noise_calibration/notebooks/*.py
poetry run jupyter notebook
```

Then, run each notebook in Jupyter.


### Reproducing Data

We provide information from experimental runs from DP-SGD in the
experiments/data folder. To reproduce these, run:

* https://github.com/ftramer/Handcrafted-DP/blob/main/scripts/run_cnns_cifar10.py
* https://github.com/microsoft/dp-transformers/blob/main/research/fine_tune_llm_w_qlora/fine-tune-nodp.py

with the parameters mentioned in the appendix of the paper. Note that the scripts above require
minor modifications to work with the parameters (i.e., adjust LoRA layers for GPT-2; by default they are set
for Mistral), and additional instrumentation to output the test accuracy metric and the mechanism parameters.
