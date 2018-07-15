# exp-anomaly-detector-AIOps
Using DNN for univariate time series anomaly detection over AIOps Competition dataset


Please log in into http://iops.ai/competition_detail/?competition_id=5&flag=1 for downloading the input files:
(unzip KPI异常检测决赛数据集.zip)
* phase2_train.csv
* phase2_ground_truth.hdf

Please check the "code" folder for details. The ipython notebook should be self-explanatory.

Hope you enjoy it.

Some take away from the experiments:
* The most critical factor that determines the result is the identification of the vital features
* The second critical factor is the scale of the features, different scale methods lead to very distinct results
* The tuning of the parameters is not as critical as expected, e.g., epoches and batch size in neural network training, and thresholds selection.
