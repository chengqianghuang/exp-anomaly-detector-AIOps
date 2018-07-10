# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import json
from sys import argv
from sklearn.metrics import f1_score


# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


# set missing = 0
def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int)
    new_label[idx] = label

    return new_label


def label_evaluation(truth_file, result_file, delay=7):
    data = {'result': False, 'data': "", 'message': ""}

    if result_file[-4:] != '.csv':
        data['message'] = "提交的文件必须是csv格式"
        return json.dumps(data, ensure_ascii=False)
    else:
        result_df = pd.read_csv(result_file)

    if 'KPI ID' not in result_df.columns or 'timestamp' not in result_df.columns or \
                    'predict' not in result_df.columns:
        data['message'] = "提交的文件必须包含KPI ID,timestamp,predict三列"
        return json.dumps(data, ensure_ascii=False)

    truth_df = pd.read_hdf(truth_file)

    kpi_names = truth_df['KPI ID'].values
    kpi_names = np.unique(kpi_names)
    y_true_list = []
    y_pred_list = []

    #print('kpi names:\n', kpi_names)
    #print('res names:\n', np.unique(result_df["KPI ID"].values))

    for kpi_name in kpi_names:

        truth = truth_df[truth_df["KPI ID"] == kpi_name]
        y_true = reconstruct_label(truth["timestamp"], truth["label"])

        if str(kpi_name) not in result_df["KPI ID"].values:
            data['message'] = "提交的文件缺少KPI %s 的结果" % kpi_name
            return json.dumps(data, ensure_ascii=False)

        result = result_df[result_df["KPI ID"] == str(kpi_name)]

        if len(truth) != len(result):
            data['message'] = "文件长度错误"
            return json.dumps(data, ensure_ascii=False)

        y_pred = reconstruct_label(result["timestamp"], result["predict"])

        y_pred = get_range_proba(y_pred, y_true, delay)
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

    try:
        fscore = f1_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
    except:
        data['message'] = "predict列只能是0或1"
        return json.dumps(data, ensure_ascii=False)

    data['result'] = True
    data['data'] = fscore
    data['message'] = '计算成功'

    return json.dumps(data, ensure_ascii=False)


if __name__ == '__main__':
    _, truth_file, result_file, delay = argv
    delay = (int)(delay)
    print(label_evaluation(truth_file, result_file, delay))

# run example:
# python evaluation.py 'ground_truth.hdf' 'predict.csv' 2
