#coding:utf-8
import numpy as np
import tensorflow as tf
import configparser
import os

class Model(object):



	def get_config(self):
		return self.config

	def get_positive_instance(self, in_batch = True):
		if in_batch:
			return [self.postive_h, self.postive_t, self.postive_r]
		else:
			return [self.batch_h[0:self.config.batch_size], \
			self.batch_t[0:self.config.batch_size], \
			self.batch_r[0:self.config.batch_size]]

	def get_negative_instance(self, in_batch = True):
		if in_batch:
			return [self.negative_h, self.negative_t, self.negative_r]
		else:
			return [self.batch_h[self.config.batch_size:self.config.batch_seq_size],\
			self.batch_t[self.config.batch_size:self.config.batch_seq_size],\
			self.batch_r[self.config.batch_size:self.config.batch_seq_size]]

	def get_all_instance(self, in_batch = False):
		if in_batch:
			return [tf.transpose(tf.reshape(self.batch_h, [1 + self.config.negative_ent + self.config.negative_rel, -1]), [1, 0]),\
			tf.transpose(tf.reshape(self.batch_t, [1 + self.config.negative_ent + self.config.negative_rel, -1]), [1, 0]),\
			tf.transpose(tf.reshape(self.batch_r, [1 + self.config.negative_ent + self.config.negative_rel, -1]), [1, 0])]
		else:
			return [self.batch_h, self.batch_t, self.batch_r]

	def get_all_labels(self, in_batch = False):
		if in_batch:
			return tf.transpose(tf.reshape(self.batch_y, [1 + self.config.negative_ent + self.config.negative_rel, -1]), [1, 0])
		else:
			return self.batch_y

	def get_predict_instance(self):
		return [self.predict_h, self.predict_t, self.predict_r]

	def input_def(self):
		config = self.config
		self.batch_h = tf.placeholder(tf.int64, [config.batch_seq_size])
		self.batch_t = tf.placeholder(tf.int64, [config.batch_seq_size])
		self.batch_r = tf.placeholder(tf.int64, [config.batch_seq_size])
		self.batch_y = tf.placeholder(tf.float32, [config.batch_seq_size])
		self.postive_h = tf.transpose(tf.reshape(self.batch_h[0:config.batch_size], [1, -1]), [1, 0])
		self.postive_t = tf.transpose(tf.reshape(self.batch_t[0:config.batch_size], [1, -1]), [1, 0])
		self.postive_r = tf.transpose(tf.reshape(self.batch_r[0:config.batch_size], [1, -1]), [1, 0])
		self.negative_h = tf.transpose(tf.reshape(self.batch_h[config.batch_size:config.batch_seq_size], [config.negative_ent + config.negative_rel, -1]), perm=[1, 0])
		self.negative_t = tf.transpose(tf.reshape(self.batch_t[config.batch_size:config.batch_seq_size], [config.negative_ent + config.negative_rel, -1]), perm=[1, 0])
		self.negative_r = tf.transpose(tf.reshape(self.batch_r[config.batch_size:config.batch_seq_size], [config.negative_ent + config.negative_rel, -1]), perm=[1, 0])
		self.predict_h = tf.placeholder(tf.int64, [None])
		self.predict_t = tf.placeholder(tf.int64, [None])
		self.predict_r = tf.placeholder(tf.int64, [None])
		self.parameter_lists = []

	def embedding_def(self):
		pass

	def loss_def(self):
		pass

	def predict_def(self):
		pass


	def __init__(self, config):

		'''revision starts'''
		if config.pretrain:

			gconfig = configparser.ConfigParser()

			curr_path = os.path.abspath(os.path.dirname(__file__))
			path = os.path.join(curr_path, "../../paths.cfg")
			gconfig.read(path)

			concept_init_embs = gconfig["paths"]["concept_vec_npy_glove"] + '.max.npy'
			relation_init_embs = gconfig["paths"]["relation_vec_npy_glove"] + '.max.npy'

			with tf.name_scope("embedding"):

				self.initial_ent_embeds = np.load(concept_init_embs)
				self.initial_rel_embeds = np.load(relation_init_embs)

		'''revision ends'''


		self.config = config

		with tf.name_scope("input"):
			self.input_def()

		with tf.name_scope("embedding"):
			self.embedding_def()

		with tf.name_scope("loss"):
			self.loss_def()

		with tf.name_scope("predict"):
			self.predict_def()