from load import num_words, X_id_test, X_id_train, Y_test, Y_train
from tensorflow.python.ops.rnn_cell import DropoutWrapper, LSTMCell, MultiRNNCell

import tensorflow as tf


if __name__ == "__main__":
    seq_length = X_id_train.shape[1]
    word_ids = tf.placeholder(tf.int32, [None, seq_length])
    # Binary classification.
    input_target = tf.placeholder(tf.int32, [None])
    target = tf.one_hot(input_target, 2)
    dropout = tf.placeholder(tf.float32)

    embedding_size = 50
    embeddings = tf.Variable(tf.truncated_normal([num_words + 1, embedding_size], stddev=0.01))
    data = tf.nn.embedding_lookup(embeddings, word_ids, validate_indices=False)

    num_neurons = 10
    num_layers = 4
    cell = LSTMCell(num_neurons)
    cell = DropoutWrapper(cell, output_keep_prob=dropout)
    cell = MultiRNNCell([cell] * num_layers)

    # Select the last output.
    output, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)

    out_size = int(target.get_shape()[1])
    weight = tf.Variable(tf.truncated_normal([num_neurons, out_size], stddev=0.01))
    bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

    cost = -tf.reduce_sum(target * tf.log(prediction))

    learning_rate = 0.01
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    optimize_model = optimizer.minimize(cost)

    num_incorrect = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(num_incorrect, tf.float32))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for epoch in range(10):
        sess.run(optimize_model, {dropout: 0.5, input_target: Y_train, word_ids: X_id_train})
        test_error = sess.run(error, {dropout: 1, input_target: Y_test, word_ids: X_id_test})
        print("Epoch {}: error {}.".format(epoch + 1, test_error))

    sess.close()

