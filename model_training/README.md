## Constructing a model to predict exome coverage from FFPE samples

Model training, and additional background are all included in the analysis notebook: build_PROCEED_model.ipynb

Below are instructions for running the notebook.

Construct conda environment to run notebook:

```
conda env create -f PROCEED.yml
```

*If having problems solving your environment with specific tool versions in PROCEED.yml, there is also a history of installation based environment named PROCEED_install_history.yml which you can use instead. Warning: this may result in slightly different tool versions.*

Enter environment:
```
conda activate PROCEED
```

Run notebook:
```
jupyter notebook build_PROCEED_model.ipynb
```
