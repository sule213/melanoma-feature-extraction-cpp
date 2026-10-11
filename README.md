# Bioimaging Statistics Notebooks

Small notebooks and scripts for the statistical analysis that sits behind bioimaging experiments: comparing groups, and reporting spread and precision correctly alongside image derived measurements.

## Contents

- `ANOVA.ipynb` - one way analysis of variance (ANOVA) for comparing measurements across groups
- `Standard Deviation.ipynb` - standard deviation: spread of measurements within a group
- `Standard Error of Mean.ipynb` - standard error of the mean (SEM): precision of the group mean
- `SEM.py` - standard error of the mean as a reusable Python script (the same calculation as the notebook)
- `plot.ipynb` - plotting measurements for figures

## How to use

Open the notebooks in Jupyter and replace the example values with your own measurements (for example, per image or per cell measurements exported from an image analysis pipeline). Each notebook is self contained and works through the calculation step by step, so it can also be used for teaching.

Requirements: Python with NumPy, Pandas, and Matplotlib, plus Jupyter for the notebooks.

## Notes

- These notebooks support the analysis in my bioimaging repositories, including [optical-flow-scratch-wound-analysis](https://github.com/sule213/optical-flow-scratch-wound-analysis).
