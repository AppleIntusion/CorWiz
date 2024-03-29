# Correlation Wizard

## Description
Generates feature-space and simplified correlation matrix heatmaps for subunits of 
molecular structures. I got the idea when reading the following [article](https://towardsdatascience.com/escape-the-correlation-matrix-into-feature-space-4d71c51f25e5). The utility was largely adapted from the [notebook](https://github.com/MattJBritton/ExquisiteCorrpse) associated with the article. It's still pretty rough around the edges.

Examples of the two types of plots that are generated are found below.

### Feature-Space Plot
![](images/feature.png)

### Correlation Heatmap
![](images/heatmap.png)

## Dependencies

I built this using python 3.6

``` python
pytraj
matplotlib
numpy
pandas
seaborn
sys
os
argparse
sklearn
scipy
```

Don't have a yml file but issuing the following commands will generate a suitable conda environment.

``` bash
$ conda create --name <your-name> python=3.6 matplotlib numpy pandas seaborn sys os argparse sklearn scipy
$ conda install -c ambermd pytraj
```

## Run Instructions

When running, activate the conda environment then issue the following:

``` python
python corr_regions.py -h
```

It will give an explaination of the arguments that are required to run the script.


