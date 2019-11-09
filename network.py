from cnn_functions import *

def network(X, hold_prob):
    """
    in shape 4,4 is the filter size i.e. we will look 4x4 everytime in the 32x32 oixem image and we will move as we defined in stride i.e. 1, 3 are the colors and 32 are the output features from each 4,4 filter
    then the 32 is coming from the previous and the 64 is again the output features from every 4x4 filter
    in 8*8*64, the 8 is coming from: the picture is 32x32 pixels so after 2 2x2 polling layers we have 32/2/2=8
    :param X: [batch,height,width,channels]
    :param the shape is refering to W in cnn)functions
    :return: y_pred
    """
    convo_1 = convolutional_layer(X, shape=[4, 4, 3, 32])
    convo_1_pooling = max_pool_2by2(convo_1)

    convo_2 = convolutional_layer(convo_1_pooling, shape=[4, 4, 32, 64])
    convo_2_pooling = max_pool_2by2(convo_2)

    convo_2_flat = tf.reshape(convo_2_pooling, [-1, 8 * 8 * 64])

    full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

    full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

    y_pred = normal_full_layer(full_one_dropout, 101)

    return y_pred


