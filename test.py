import numpy
import sonnet as snt
import tensorflow as tf
import logging


class MyMLP(snt.Module):
    def __init__(self, name=None):
        super(MyMLP, self).__init__(name=name)
        self.hidden1 = snt.Linear(1024, name="hidden1")
        self.output = snt.Linear(10, name="output")

    def __call__(self, x):
        x = self.hidden1(x)
        x = tf.nn.relu(x)
        x = self.output(x)
        return x

    def get_all_variables(self):
        return super(MyMLP, self).trainable_variables


class MyRNNCore(snt.RNNCore):
    def __init__(self, nh_lstm, nh_bottleneck, name="test_rnn_core"):
        super(MyRNNCore, self).__init__(name=name)
        self._nh_lstm = nh_lstm
        self._nh_bottleneck = nh_bottleneck
        self._lstm = snt.LSTM(self._nh_lstm)
        self._linear = snt.Linear(self._nh_bottleneck)

    def initial_state(self, batch_size, **kwargs):
        hidden = tf.zeros(shape=(batch_size, self._nh_lstm))
        cell = tf.zeros(shape=(batch_size, self._nh_lstm))
        return snt.LSTMState(hidden, cell)

    def __call__(self, inputs, prev_state):
        conc_inputs = tf.concat(inputs, axis=1, name="conc_inputs")  # shape ( ,BN)
        lstm_inputs = conc_inputs
        lstm_output, next_state = self._lstm(lstm_inputs, prev_state)
        bottleneck = self._linear(lstm_output)
        # return (bottleneck, lstm_output), tuple(list(next_state))
        return (lstm_output, bottleneck), next_state


class MyRNN(snt.Module):
    def __init__(self, rnn_cell, nh_lstm, name="grid_cell_supervised"):
        super(MyRNN, self).__init__(name=name)
        self._core = rnn_cell  # here lstm_cell
        self._nh_lstm = nh_lstm  # Size of LSTM cell.

    def __call__(self, input_sequence, training=False):
        output_seq, final_state = snt.dynamic_unroll(core=rnn_cell,
                                                     input_sequence=input_sequence,
                                                     initial_state=rnn_cell.initial_state(batch))
        return output_seq, final_state


if __name__ == '__main__':
    logging.basicConfig(filename='/home/kejia/grid-cells/log/example.log', level=logging.DEBUG)
    logging.debug('This message should go to the log file')
    logging.info('So should this')
    logging.warning('And this, too')
    logging.error('And non-ASCII stuff, too, like Øresund and Malmö')

    mlp1 = MyMLP()
    mlp_input = tf.random.normal([8, 28 * 28])
    # print(mlp_input)
    mlp_target = [0.2, 0.3, 0.5, 0.4, 0.3, 0.4, 0.5, 0.9, 0.2, 0.6]
    mlp_target = tf.reshape(mlp_target, [1, 10])
    # mlp2 = MyMLP()
    # output1 = mlp1(tf.random.normal([8, 28 * 28]))
    # output2 = mlp2(tf.random.normal([5, 28 * 28]))
    # print('output', output1)
    # variables1 = mlp1.get_all_variables() # tuple(4)
    # variables2 = mlp1.get_all_variables() # tuple(4)
    # variables = variables1 + variables2 # tuple(8)
    # hidden1_b, hidden1_w, output_b, output_w = variables1
    # regularizer_keras = tf.keras.regularizers.l2(l2=0.01)
    # # regularizer_snt = snt.regularizers.L2(0.01)
    # regular_loss_keras = regularizer_keras(hidden1_w)
    # hidden1_variables = hidden1_b + hidden1_w
    # print(tf.shape(hidden1_variables))
    # # regular_loss_snt = regularizer_snt(tf.constant(hidden1_w))
    # print(variables)

    # t1 = [[1, 2, 3], [4, 5, 6]] # shape B=2, N=3
    # print(tf.concat(t1,0))
    batch = 2
    input_sequence = tf.random.uniform([1, batch, 3])
    # print(input_sequence)
    t1 = [[0.1, 0.2], [0.4, 0.5]]
    target_sequence = tf.expand_dims(t1, 0)
    rnn_cell = MyRNNCore(4, 2)
    # print(rnn_cell.initial_state(batch_size=batch))
    rnn_layer = MyRNN(rnn_cell, 2)
    # simple running test
    # output_seq, final_state = rnn_layer(input_sequence=input_sequence)
    # print("output", output_seq)  # shape = (sequence*batch*nh_lstm) => ((sequence*batch*nh_lstm), (sequence*batch*nh_bottleneck))
    # print("final_state", final_state)  # [h,c] = [(batch*nh_lstm), (batch*nh_lstm)]

    # lstm_cell = snt.LSTM(4)
    # vanilla_cell = snt.VanillaRNN(4)
    # print(lstm_cell.initial_state(2))
    # print(vanilla_cell.initial_state(2))

    # run with training process

    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # without tf.function
    # def train_rnn_step(input_seq, target_seq):
    #     print(input_seq)
    #     with tf.GradientTape() as tape:
    #         output_seq, final_state = rnn_layer(input_sequence=input_seq)
    #         output = tf.reshape(output_seq[0], [4, 1])
    #         # regularization_loss = tf.math.add_n(model.losses)
    #         loss = loss_fn(target_seq, output)
    #         print('training loss', loss)
    #         # total_loss = pred_loss + regularization_loss

    @tf.function
    def train_mlp_step(input, target):
        with tf.GradientTape() as tape:
            predictions = mlp1(input)
            print("mlp prediction", predictions)
            loss = loss_fn(target, tf.transpose(predictions))
            print('training loss', loss)

        gradients = tape.gradient(loss, mlp1.trainable_variables)
        optimizer.apply_gradients(zip(gradients, mlp1.trainable_variables))

    # @tf.function
    def train_rnn_step(input_seq, target_seq):
        print(input_seq)
        with tf.GradientTape() as tape:
            output_seq, final_state = rnn_layer(input_sequence=input_seq)
            print(output_seq)  # shape = ([1,2,4], [1,2,2])
            output = tf.reshape(output_seq[1], [4, 1])
            # regularization_loss = tf.math.add_n(model.losses)
            loss = loss_fn(target_seq, output)
            print('training loss', loss)
            # total_loss = pred_loss + regularization_loss

        print("trainable variables", rnn_layer.trainable_variables)
        gradients = tape.gradient(loss, rnn_layer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, rnn_layer.trainable_variables))
        return loss


    for epoch in range(5):
        # training_loss = train_mlp_step(mlp_input, mlp_target)
        training_loss = train_rnn_step(input_sequence, target_sequence)
        print("Finished epoch", epoch)
