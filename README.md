# Not So Sudden Death - Death Prediction in CS:GO
Code used in my master's thesis to enable real-time prediction of in-game deaths in CS:GO.

## Data
You can collect your own raw CS:GO match data from hltv.org by downloading the match demo files directly from this site. They are numbered sequentially (i.e 61234.zip, 61235.zip) so it's trivial to generate a list of files and download them in order, though 1 at a time to avoid throttling.
Alternatively, you can use my kaggle datasets:

https://www.kaggle.com/datasets/stefan8888/prediction-of-ingame-deaths-in-csgo-raw (Not windowed) - Preferable if you want to do your own windowing (or none at all). use this data from step 4 and onward in preprocessing.ipynb  
https://www.kaggle.com/datasets/stefan8888/prediction-of-ingame-deaths-in-csgo (ML-Ready: Windowed with 5 steps) - Preferable if you want to use my exact data to compare your model performance. All steps in preprocessing.ipynb were already completed

## Notebooks
preprocessing.ipynb - ETL, data cleaning, feature generation, OHE and splitting the data to make it ready for ML  
models.ipynb - Building the 3 ML models and comparing performance  
visualizing.ipynb - visually evaluating performance of the best model on a selected match/round  

Hopefully, the utilization of the .py files becomes obvious as you go through the notebooks.

## Future work
Better model performance to increase potential usability of the model as an eSports storytelling aid  
More specific framework on evaluating predictions in their usefulness for eSports usecase rather acc/pr/f1 etc

## Acknowledgements
This dataset was made possible through the awpy package. ***[Xenopoulos, Peter, et al. "Valuing Actions in Counter-Strike: Global Offensive." 2020 IEEE International Conference on Big Data (Big Data). IEEE, 2020]*** - https://github.com/pnxenopoulos/awpy
