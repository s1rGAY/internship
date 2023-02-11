dvc init
dvc remote add --default 'PredFeatureSales' 'mega:/PredFeatureSales'
dvc add 'mega:/PredFeatureSales'
dvc push
dvc pull