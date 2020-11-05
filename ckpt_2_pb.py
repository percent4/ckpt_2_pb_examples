# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python import saved_model

export_path = "pb_models/lr/1"

graph = tf.Graph()
saver = tf.train.import_meta_graph("./model/lr.ckpt.meta", graph=graph)
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint("./model"))
    saved_model.simple_save(session=sess,
                            export_dir=export_path,
                            inputs={"x": graph.get_operation_by_name('x').outputs[0]},
                            outputs={"y_pred": graph.get_operation_by_name('inference/y_pred').outputs[0]})


'''
> saved_model_cli show --dir 1 --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 3)
        name: x:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['y_pred'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, -1)
        name: inference/y_pred:0
  Method name is: tensorflow/serving/predict
'''

