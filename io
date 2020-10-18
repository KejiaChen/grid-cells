
reader = <ShuffleDataset shapes: {ego_vel: (100, 3), init_hd: (1,), init_pos: (2,), target_hd: (100, 1), target_pos: (100, 2)}, types: {ego_vel: tf.float32, init_hd: tf.float32, init_pos: tf.float32, target_hd: tf.float32, target_pos: tf.float32}>
test = 

reader_batch = <BatchDataset shapes: {ego_vel: (None, 100, 3), init_hd: (None, 1), init_pos: (None, 2), target_hd: (None, 100, 1), target_pos: (None, 100, 2)}, types: {ego_vel: tf.float32, init_hd: tf.float32, init_pos: tf.float32, target_hd: tf.float32, target_pos: tf.float32}>





<class 'tensorflow.python.framework.ops.Tensor'>
Tensor("random_shuffle_queue_DequeueMany:0", shape=(10, 2), dtype=float32)
range ((-1.1, 1.1), (-1.1, 1.1))


example_dict = {'target_pos': <tf.Tensor 'ParseExample/ParseExample:4' shape=(?, 100, 2) dtype=float32>, 'target_hd': <tf.Tensor 'ParseExample/ParseExample:3' shape=(?, 100, 1) dtype=float32>, 'ego_vel': <tf.Tensor 'ParseExample/ParseExample:0' shape=(?, 100, 3) dtype=float32>, 'init_pos': <tf.Tensor 'ParseExample/ParseExample:2' shape=(?, 2) dtype=float32>, 'init_hd': <tf.Tensor 'ParseExample/ParseExample:1' shape=(?, 1) dtype=float32>}

batch = [<tf.Tensor 'ParseExample/ParseExample:2' shape=(?, 2) dtype=float32>, <tf.Tensor 'ParseExample/ParseExample:1' shape=(?, 1) dtype=float32>, <tf.Tensor 'strided_slice:0' shape=(?, 100, 3) dtype=float32>, <tf.Tensor 'strided_slice_1:0' shape=(?, 100, 2) dtype=float32>, <tf.Tensor 'strided_slice_2:0' shape=(?, 100, 1) dtype=float32>]

read_ops=[[<tf.Tensor 'ParseExample/ParseExample:2' shape=(?, 2) dtype=float32>, <tf.Tensor 'ParseExample/ParseExample:1' shape=(?, 1) dtype=float32>, <tf.Tensor 'strided_slice:0' shape=(?, 100, 3) dtype=float32>, <tf.Tensor 'strided_slice_1:0' shape=(?, 100, 2) dtype=float32>, <tf.Tensor 'strided_slice_2:0' shape=(?, 100, 1) dtype=float32>], [<tf.Tensor 'ParseExample_1/ParseExample:2' shape=(?, 2) dtype=float32>, <tf.Tensor 'ParseExample_1/ParseExample:1' shape=(?, 1) dtype=float32>, <tf.Tensor 'strided_slice_3:0' shape=(?, 100, 3) dtype=float32>, <tf.Tensor 'strided_slice_4:0' shape=(?, 100, 2) dtype=float32>, <tf.Tensor 'strided_slice_5:0' shape=(?, 100, 1) dtype=float32>], [<tf.Tensor 'ParseExample_2/ParseExample:2' shape=(?, 2) dtype=float32>, <tf.Tensor 'ParseExample_2/ParseExample:1' shape=(?, 1) dtype=float32>, <tf.Tensor 'strided_slice_6:0' shape=(?, 100, 3) dtype=float32>, <tf.Tensor 'strided_slice_7:0' shape=(?, 100, 2) dtype=float32>, <tf.Tensor 'strided_slice_8:0' shape=(?, 100, 1) dtype=float32>], [<tf.Tensor 'ParseExample_3/ParseExample:2' shape=(?, 2) dtype=float32>, <tf.Tensor 'ParseExample_3/ParseExample:1' shape=(?, 1) dtype=float32>, <tf.Tensor 'strided_slice_9:0' shape=(?, 100, 3) dtype=float32>, <tf.Tensor 'strided_slice_10:0' shape=(?, 100, 2) dtype=float32>, <tf.Tensor 'strided_slice_11:0' shape=(?, 100, 1) dtype=float32>]]


train_trajectory= (<tf.Tensor 'random_shuffle_queue_DequeueMany:0' shape=(10, 2) dtype=float32>, <tf.Tensor 'random_shuffle_queue_DequeueMany:1' shape=(10, 1) dtype=float32>, <tf.Tensor 'random_shuffle_queue_DequeueMany:2' shape=(10, 100, 3) dtype=float32>, <tf.Tensor 'random_shuffle_queue_DequeueMany:3' shape=(10, 100, 2) dtype=float32>, <tf.Tensor 'random_shuffle_queue_DequeueMany:4' shape=(10, 100, 1) dtype=float32>)
