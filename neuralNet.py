# Imports
import numpy as np
import tensorflow as tf

pic_size = 128*128

output_nodes = 10
nodes_hl_1 = 300
nodes_hl_2 = 300
nodes_hl_3 = 300

def model(data):
	hl_1 = {"weight": tf.Variable(tf.random_normal([pic_size, nodes_hl_1])),
			"bias": tf.Variable(tf.random_normal(nodes_hl_1))}

	hl_2 = {"weight": tf.Variable(tf.random_normal([nodes_hl_1, nodes_hl_2])),
			"bias": tf.Variable(tf.random_normal(nodes_hl_2))}

	hl_3 = {"weight": tf.Variable(tf.random_normal([nodes_hl_2, nodes_hl_3])),
			"bias": tf.Variable(tf.random_normal(nodes_hl_3))}

	ol = {"weight": tf.Variable(tf.random_normal([nodes_hl_3, output_nodes])),
			"bias": tf.Variable(tf.random_normal(output_nodes))}

	l1 = tf.add(tf.matmul(data, hl_1["weight"]), hl_1["bias"])
	l1 = tf.nn.relu(l1)

	l1 = tf.add(tf.matmul(l1, hl_2["weight"]), hl_2["bias"])
	l1 = tf.nn.relu(l1)

	l1 = tf.add(tf.matmul(l1, hl_3["weight"]), hl_3["bias"])
	l1 = tf.nn.relu(l1)

	output = tf.add(tf.matmul(l1, ol["weight"]), ol["bias"])

	return output

def train