# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.summary import FileWriter

sess = tf.Session()
tf.train.import_meta_graph("./model/lr.ckpt.meta")
FileWriter("logs/1", sess.graph)
sess.close()