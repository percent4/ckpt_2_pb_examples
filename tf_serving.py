# -*- coding: utf-8 -*-
import requests

# 利用tensorflow/serving的HTTP接口请求进行预测
x = [1, 2, 2]
tensor = {"instances": [{"x": x}]}

url = "http://192.168.1.193:8551/v1/models/lr:predict"
req = requests.post(url, json=tensor)
print(req.status_code)
y_pred = req.json()['predictions'][0]
print(y_pred)

# 直接按照tf模型训练的系数进行预测
w = [0.3013742, 0.5024621, 0.09835612]
b = -0.19986232
result = 0
for data, weight in zip(x, w):
    result += data*weight

result += b
print(result)