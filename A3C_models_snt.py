import sonnet as snt
from A3C_utils import *


class ConvNet(snt.Module):
    def __init__(self,  **conv_kwarg):
        super(ConvNet, self).__init__()
        self.conv_layer1 = snt.Conv2D(16, 5, stride=2, padding='same', name="conv1")
        self.conv_layer2 = snt.Conv2D(32, 5, stride=2, padding='same', name="conv2")
        self.conv_layer3 = snt.Conv2D(64, 5, stride=2, padding='same', name="conv3")
        self.conv_layer4 = snt.Conv2D(128, 5, stride=2, padding='same', name="conv4")
        self.bottleneck_layer = snt.Dense(self._nh_bottleneck, name="bottleneck")

    def call(self, frame):
        """
        Args:
            frame: visual inputs list (1, 63, 64, 3), with values in [-1,1]
        """
        frame = tf.cast(frame, tf.float32)
        shape = frame._shape_as_list()
        # frame = tf.reshape(frame, (shape[0] * shape[1], shape[2], shape[3], shape[4]))
        conv_out = frame  # (N, W, H, C)

        # frame /= 255  # [-1,1]
        # conc_inputs = tf.concat(inputs, axis=1, name="conc_inputs")  # shape ( ,BN)

        # convnet
        conv_out = self.conv_layer1(conv_out)
        conv_out = tf.nn.relu(conv_out)
        conv_out = self.conv_layer2(conv_out)
        conv_out = tf.nn.relu(conv_out)
        conv_out = self.conv_layer3(conv_out)
        conv_out = tf.nn.relu(conv_out)
        conv_out = self.conv_layer4(conv_out)
        conv_out = tf.nn.relu(conv_out)

        # conv_out = tf.reshape(conv_out, (shape[0] * shape[1], -1))
        # for layer in self.conv_net:
        #     conv_out = layer[conv_out]
        #     conv_out = tf.nn.relu(conv_out)

        bottleneck_output = self.bottleneck_layer(conv_out)
        return bottleneck_output


class MLP(snt.Module):
    def __init__(self, **conv_kwarg):
        super(MLP, self).__init__()
        self.fc1 = snt.Linear(128)
        self.fc2 = snt.Linear(256)
        self.fc3 = snt.Linear(256)

    def call(self, x):
        x = self.fc1(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.fc2(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.fc3(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x

# class ACModel(tf.keras.Model):
#     """Network Structure"""
#     def __init__(self, num_actions, num_hidden_units):
#         super(ACModel, self).__init__()
#         self.common= ConvNet()
#         self.lstm = layers.LSTMCell(num_hidden_units)  # activation="relu")
#         self.actor = layers.Dense(num_actions, activation="softmax")
#         self.critic = layers.Dense(1)
#
#     def call(self, inputs):
#         x, (ht, ct) = inputs
#         common_output
#         x, (ht, ct) = self.lstm(conv_output, states=[ht, ct])
#         return self.actor(x), self.critic(x), (ht, ct)


