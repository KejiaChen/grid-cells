import tensorflow as tf
import tensorflow.keras as keras

# class Mycell(keras.layers.Layer):
#     def __init__(self, lstm_units, bottleneck_units, **kwargs):
#         super(Mycell, self).__init__()
#         self.lstm_units = lstm_units
#         self.state_size = lstm_units
#         self.bottleneck_units = bottleneck_units
#         self.lstm = keras.layers.LSTMCell(units=self.lstm_units)
#         self.linear1 = keras.layers.Dense(units=self.bottleneck_units)
#
#     # def build(self, input_shape):
#     #     self.lstm = keras.layers.LSTMCell(units=self.lstm_units)
#     #     self.linear1 = keras.layers.Dense(units=self.bottleneck_units)
#
#     def __call__(self, inputs, prev_state):
#         lstm_output, next_state = self.lstm(inputs, prev_state)
#         bottleneck = self.linear1(lstm_output)
#         return bottleneck

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        self.kernel = keras.layers.LSTMCell(units=self.units)
        # self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
        #                               initializer='uniform',
        #                               name='kernel')
        # self.recurrent_kernel = self.add_weight(
        #     shape=(self.units, self.units),
        #     initializer='uniform',
        #     name='recurrent_kernel')
        # self.built = True

    def __call__(self, inputs, states):
        # prev_output = states[0]
        # h = keras.backend.dot(inputs, self.kernel)
        # output = h + keras.backend.dot(prev_output, self.recurrent_kernel)
        output, next_state = self.kernel(inputs, states)
        return output, [output]

if __name__ == '__main__':
    # cell = MinimalRNNCell(32) # output_size
    cell = keras.layers.LSTMCell(32)
    print(cell.state_size)
    print(cell.output_size)
    initial = tf.zeros([32, 32])
    x = keras.Input((10,5)) # batch_size=none
    print(x)
    # output, next_state = cell(x,initial)
    layer = keras.layers.RNN(cell)
    y = layer(x)

    # inputs = tf.random.normal([32, 10, 8])
    # 32: batch_size; 10: sequence_length; 8:input_size
    # rnn_cell = keras.layers.LSTMCell(4)
    # 4 :output_size(i.e. untis)
    # rnn = keras.layers.RNN(rnn_cell)
    # output = rnn(inputs)

    # rnn = tf.keras.layers.RNN(
    #     rnn_cell,
    #     return_sequences=True,
    #     return_state=True)
    # whole_seq_output, final_memory_state, final_carry_state = rnn(inputs)

    print(y)
