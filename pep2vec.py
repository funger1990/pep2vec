#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import argparse
import itertools
import gzip
import re

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# ----------------------------------------------
class Pep2Vec(object):
    """Word2Vec model (Skipgram)."""

    def __init__(self, args, sess):
        self.args = args
        self.sess = sess
        self.fin = False
        self.num_samples = args.batch_size * args.negative_sample
        self.beg_time = time.time()
        self.log = os.path.join(args.save_path, 'log')

        self.create_vocab()
        self.load_context()
        self.build_graph()

    def create_vocab(self):
        alphabet = list('ACDEFGHIKLMNPQRSTVWY')
        id2word = [''.join(i) for i in itertools.product(alphabet, repeat=self.args.pep_len)]
        word2id = dict(zip(id2word, xrange(len(id2word))))
        self.vocab_size = len(id2word)
        self.word2id = word2id

    def load_context(self):
        context_counter = {}
        with open(self.args.context_file) as f:
            for line in f:
                word, cnt = line.strip().split('\t')
                context_counter[word] = int(cnt)
        
        id2context = sorted(context_counter.keys())
        context2id = dict(zip(id2context, xrange(len(id2context))))
        context_counts = [context_counter[id2context[i]] for i in xrange(len(context_counter))]
        total_sample = sum(context_counts)

        self.context_size = len(id2context)
        self.context2id = context2id
        self.context_counts = context_counts
        self.total_sample = total_sample

    def generate_batch(self):
        # open gzipped file
        if not self.fin:
            self.fin = gzip.open(self.args.train_data)

        full_batch = True
        batch_example, batch_label1, batch_label2 = [], [], []
        for _ in xrange(self.args.batch_size):
            line = self.fin.readline()
            item = line.strip().split()
            # < batch size
            if not item:
                full_batch = False
                break
            # in order of NMC, word => id
            batch_example.append(self.word2id[item[1]])
            batch_label1.append(self.context2id[item[0]])
            batch_label2.append(self.context2id[item[2]])

        # closefile
        if not full_batch:
            self.fin.close()
            self.fin = False
            batch_example, batch_label1, batch_label2 = None, None, None

        return batch_example, batch_label1, batch_label2

    def forward(self, example, label):
        """Build the graph for the forward pass."""
        # word2vec
        opts = self.args

        # Declare all variables we need.
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / opts.emb_dim
        embedding = tf.Variable(
            tf.random_uniform([self.vocab_size, opts.emb_dim], -init_width, init_width),
            name="emb")

        # full-connected weight: [context_size, emb_dim]. Transposed.
        fc_w_t = tf.Variable(
            tf.zeros([self.context_size, opts.emb_dim]),
            name="fc_w_t")
        # bias: [context_size].
        fc_b = tf.Variable(tf.zeros([self.context_size]), name="fc_b")

        # Negative sampling.
        negative, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.reshape(tf.cast(label, tf.int64),    # must be tf.int64
                                    [opts.batch_size, 1]),
            num_true=1,
            num_sampled=self.num_samples,
            unique=True,
            range_max=self.context_size,
            distortion=0.75,
            unigrams=self.context_counts
        )

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(embedding, example)

        # Weights for labels: [batch_size, emb_dim]
        label_w_t = tf.nn.embedding_lookup(fc_w_t, label)
        # Biases for labels: [batch_size, 1]
        label_b = tf.nn.embedding_lookup(fc_b, label)

        # Weights for sampled ids: [num_sampled, emb_dim]
        negative_w_t = tf.nn.embedding_lookup(fc_w_t, negative)
        # Biases for sampled ids: [num_sampled, 1]
        negative_b = tf.nn.embedding_lookup(fc_b, negative)

        # True logits: [batch_size, 1]
        positive_logits = tf.matmul(example_emb, label_w_t, transpose_b=True) + \
                          tf.reshape(label_b, [opts.batch_size])

        # Sampled logits: [batch_size, num_sampled * 2]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        negative_logits = tf.matmul(example_emb, negative_w_t, transpose_b=True) + \
                          tf.reshape(negative_b, [self.num_samples])

        return positive_logits, negative_logits

    def forward_nc(self, example, label1, label2):
        """Build the graph for the forward pass."""
        # N context and C context
        opts = self.args

        # Declare all variables we need.
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / opts.emb_dim
        embedding = tf.Variable(
            tf.random_uniform([self.vocab_size, opts.emb_dim], -init_width, init_width),
            name="emb")

        # N terminal + C terminal
        # full-connected weight: [context_size, emb_dim]. Transposed.
        fc_w_t = tf.Variable(
            tf.zeros([self.context_size * 2, opts.emb_dim]),
            name="fc_w_t")
        # bias: [context_size].
        fc_b = tf.Variable(tf.zeros([self.context_size * 2]), name="fc_b")

        # Negative sampling.
        negative1, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.reshape(tf.cast(label1, tf.int64),    # must be tf.int64
                                    [opts.batch_size, 1]),
            num_true=1,
            num_sampled=self.num_samples,
            unique=True,
            range_max=self.context_size,
            distortion=0.75,
            unigrams=self.context_counts
        )

        negative2, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.reshape(tf.cast(label2, tf.int64),
                                    [opts.batch_size, 1]),
            num_true=1,
            num_sampled=self.num_samples,
            unique=True,
            range_max=self.context_size,
            distortion=0.75,
            unigrams=self.context_counts
        )

        # merge N negative samples and C negative samples: [batch_size * 2]
        example = tf.concat([example, example], axis=0)
        label = tf.concat([label1, label2 + self.context_size], axis=0)
        negative = tf.concat([negative1, negative2 + self.context_size], axis=0)

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(embedding, example)

        # Weights for labels: [batch_size * 2, emb_dim]
        label_w_t = tf.nn.embedding_lookup(fc_w_t, label)
        # Biases for labels: [batch_size * 2, 1]
        label_b = tf.nn.embedding_lookup(fc_b, label)

        # Weights for sampled ids: [num_sampled * 2, emb_dim]
        negative_w_t = tf.nn.embedding_lookup(fc_w_t, negative)
        # Biases for sampled ids: [num_sampled * 2, 1]
        negative_b = tf.nn.embedding_lookup(fc_b, negative)

        # True logits: [batch_size, 1]
        positive_logits = tf.matmul(example_emb, label_w_t, transpose_b=True) + \
                          tf.reshape(label_b, [opts.batch_size * 2])

        # Sampled logits: [batch_size, num_sampled * 2]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        negative_logits = tf.matmul(example_emb, negative_w_t, transpose_b=True) + \
                          tf.reshape(negative_b, [self.num_samples * 2])

        return positive_logits, negative_logits

    def nce_loss(self, positive_logits, negative_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        positive_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(positive_logits), logits=positive_logits)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(negative_logits), logits=negative_logits)

        # NCE-loss is the sum of the true and noise (negative sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(positive_xent) + tf.reduce_sum(negative_xent)) / self.args.batch_size

        return nce_loss_tensor

    def optimize(self, loss):
        """Build the graph to optimize the loss function."""
        # Optimizer nodes.

        # Linear learning rate decay.
        # tensor, shape []
        ratio_trained_words = (tf.cast(self._global_step, tf.float32) * self.args.batch_size) / \
                              (self.args.epochs_to_train * self.total_sample)
        lr = self.args.learning_rate * tf.maximum(
            0.0001,
            1.0 - ratio_trained_words)

        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_step = optimizer.minimize(loss, global_step=self._global_step)

        return train_step, lr

    def build_graph(self):
        """Build the graph for the full model."""
        # scalar tensor, shape []
        self._epoch = tf.Variable(0, name='epoch')
        self._increment_epoch = tf.assign_add(self._epoch, 1)

        self._time = tf.Variable(0, dtype=tf.float32, name='time')
        self._increment_time = lambda x: tf.assign_add(self._time, x)

        self._global_step = tf.Variable(0, name="global_step")

        # input
        self._example = tf.placeholder(tf.int32, shape=[self.args.batch_size])
        self._label1 = tf.placeholder(tf.int32, shape=[self.args.batch_size])
        self._label2 = tf.placeholder(tf.int32, shape=[self.args.batch_size])

        # N C context
        positive_logits, negative_logits = self.forward_nc(self._example, self._label1, self._label2)

        # word2vec
        # positive_logits, negative_logits = self.forward(example, label1)
        # positive_logits_2, negative_logits_2 = self.forward(example, label2)
        # positive_logits = positive_logits + positive_logits_2
        # negative_logits = negative_logits + negative_logits_2

        self._loss = self.nce_loss(positive_logits, negative_logits)
        self._train_step, self._lr = self.optimize(self._loss)

        tf.summary.scalar('global_step', self._global_step)
        tf.summary.scalar("NCE_loss", self._loss)
        tf.summary.scalar('learning_rate', self._lr)
        self._merged = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(self.args.save_path, self.sess.graph)
        self.saver = tf.train.Saver()

        # Properly initialize all variables.
        tf.global_variables_initializer().run()

    def train(self):
        """Train the model."""
        epoch = self.sess.run(self._epoch)
        last_time = self.sess.run(self._time)
        beg_time = time.time()

        while True:
            batch_input, batch_label1, batch_label2 = self.generate_batch()
            if batch_input is None:
                break

            feed_dict = {self._example: batch_input, self._label1: batch_label1, self._label2: batch_label2}
            # feed_dict = {self._example: batch_input, self._label1: batch_label1}
            
            _, step, loss, lr, summary = self.sess.run(
                [self._train_step, self._global_step, self._loss, self._lr, self._merged],
                feed_dict=feed_dict)

            if step % self.args.summary_interval == 0:
                self.writer.add_summary(summary, step)

                current_time = last_time + (time.time() - beg_time) / 60
                to_write = "Epoch %4d, Step %8d, lr = %5.3f, loss = %6.2f, time = %.1f" % \
                           (epoch + 1, step, lr, loss, current_time)
                print(to_write)
                sys.stdout.flush()
                with open(self.log, 'a') as f:
                    f.write(to_write + '\n')

        self.sess.run(self._increment_epoch)
        self.sess.run(self._increment_time((time.time() - beg_time) / 60))


# --------------------------------------------------------------
def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path",
                        # required=True,
                        default='/mnt/g/data/uniprot/tf_model/pep3_n3c3_v40_e4_lr005',
                        help="Directory to write the model and training summaries."
                        )
    parser.add_argument("--train_data",
                        # required=True,
                        default='/mnt/g/data/uniprot/corpus/uniprot_sprot_pep2vec_n3_m3_c3.txt.gz',
                        help="Training text file."
                        )
    parser.add_argument('--pep_len',
                        default=3,
                        help='length of peptide as vocabulary')
    parser.add_argument('--context_file',
                        default='/mnt/g/data/uniprot/corpus/uniprot_sprot_vocab_k3.tsv'
                        )
    parser.add_argument("--emb_dim",
                        type=int,
                        default=40,
                        help="The embedding dimension size."
                        )
    parser.add_argument("--epochs_to_train",
                        type=int,
                        default=4,
                        help="Number of epochs to train. Each epoch processes the training data once completely."
                             "the learning rate decays linearly to zero and the training stops."
                        )
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.05,
                        help="Initial learning rate."
                        )
    parser.add_argument("--negative_sample",
                        type=int,
                        default=8,
                        help="Negative samples per training example."
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="Number of training examples processed per step (size of a minibatch)."
                        )
    parser.add_argument("--summary_interval",
                        type=int,
                        default=10 ** 5,
                        help="Save training summary to file every n steps"
                        )

    args = parser.parse_args()

    return args

def main():
    """Train a word2vec model."""
    # parse arguments
    args = parse_argument()

    # create directory to write out summaries
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with tf.Graph().as_default(), tf.Session() as sess:
        # with tf.device("/cpu:0"):
        model = Pep2Vec(args, sess)

        # load latest checkpoint if any
        if os.path.exists(os.path.join(args.save_path, 'checkpoint')):
            print('Resume training!')
            saver = tf.train.import_meta_graph(args.save_path)
            saver.restore(sess, tf.train.latest_checkpoint(args.save_path))
            epoch_trained = model.sess.run(model._epoch)
            print(model.sess.run([model._epoch, model._global_step, model._lr, model._loss, model._time]))
        else:
            print('Start training!')
            epoch_trained = 0

        for epoch in xrange(args.epochs_to_train - epoch_trained):
            model.train()  # Process one epoch
            # save every epoch
            model.saver.save(sess,
                             os.path.join(args.save_path, "model.ckpt"),
                             global_step=model._global_step)

def save_embedding(outdir):
    infile_list = []
    for fname in os.listdir(outdir):
        if fname.endswith('.meta'):
            step = int(re.search(r'-(\d+)\.meta', fname).group(1))
            infile = os.path.join(outdir, fname)
            infile_list.append((step, infile))
    infile_list.sort()

    for i, (_, infile) in enumerate(infile_list):
        outfile = os.path.join(outdir, 'embedding_{}.npy'.format(i + 1))
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(infile)
            # saver.restore(sess, tf.train.latest_checkpoint(outdir))
            saver.restore(sess, infile.replace('.meta', ''))
            vec = sess.run('emb:0')
            np.save(outfile, vec)


# ------------------------------------------------------------
if __name__ == "__main__":
    main()

    # outdir = '/mnt/g/data/uniprot/tf_model/pep2_n3c3_v40_e4_lr02'
    # save_embedding(outdir)


