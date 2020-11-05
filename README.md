将ckpt转化为pb文件的一个例子

- 利用tensorflow进行模型训练

```
python linear_regression.py
```

- 查看ckpt模型图结构

```
python view_ckpt.py
tensorboard --logdir=logs/1
```

- 将ckpt文件转换为pb文件

```
python ckpt_2_pb.py
```

调用saved_model命令查看pb模型的结构（输入、输出）：

> saved_model_cli show --dir 1 --all

- 利用tensorflow/serving部署pb文件

```
docker run -t --rm -p 8551:8501 -v "path_to_pb_models/pb_models/lr:/models/lr" -e MODEL_NAME=lr tensorflow/serving
```

- 调用tensorflow/serving的HTTP接口进行预测

```
python tf_serving.py
```