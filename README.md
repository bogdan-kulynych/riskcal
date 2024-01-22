# Risk Calibration for Differential Privacy

## Installing Dependencies for Reproducing

```
poetry install --with experiments --with dev
```

## Reproducing plots

```
poetry run jupytext --to ipynb experiments/notebooks/*.py
poetry run jupyter notebook
```

Then, run the notebooks.
