#import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
#from sklearn.preprocessing import minmax_scale
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
#from keras.optimizers import adam_v2
#import torch
#import torch.nn as nn
import config

DROPOUT = config.DROPOUT

'''
def MNN_torch(data, label):
    input_size = list(data.shape)[1]
    learning_rate = 0.01
    output_size = len(label)

    class LinearClassifier(nn.Module):
        def __init__(self):
            super.__init__(self)
            self.model = nn.Sequential(
                nn.Linear(151, 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, 3),
                nn.Softmax(dim=1)
            )

        def forward(self, batch):
            y = self.model(batch)
            return y

        def optimization(self):
            self.optimizer = nn.NLLLoss(self.model.parameters())
            return self.optimizer

    model = LinearClassifier(data, label)
    # define execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device\n")
    model.to(device)
    # learn
    #for epoch in range(10):
'''



def MNN_keras(data, label, hidden_layer, node_num):
    # reference : https://iostream.tistory.com/111
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=20)

    input_num = len(X_train[0])
    output_num = 3

    # np array
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # label One-hot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # MLP
    model = Sequential()

    # Input Layer
    model.add(Dense(node_num, input_dim=input_num, kernel_initializer='glorot_uniform', activation='relu'))
    #model.add(Dropout(0.2))

    # Hidden Layer
    for i in range(hidden_layer):
        model.add(Dense(node_num, kernel_initializer='glorot_uniform', activation='relu'))
        if DROPOUT == 1:
            model.add(Dropout(0.3))

    # Output Layer
    model.add(Dense(output_num, activation='softmax'))

    # cost function & optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model training
    training_epochs = 10
    batch_size = 4
    model.fit(X_train, y_train, epochs=training_epochs, batch_size=batch_size)

    # model evaluation using test set
    print("evaluation")
    evaluation = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Accuracy: ' + str(evaluation[1]))
    model.save('beekeeping.h5')

    return evaluation[1]

'''
def MNN_tensorflow(input_num, X, Y):
    W1 = tf.Variable(tf.random_normal([input_num, 30]), name='weight1')
    b1 = tf.Variable(tf.random_normal([30]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.random_mormal([30, 30]), name='weight2')
    b2 = tf.Variable(tf.random_mormal([30]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    W3 = tf.Variable(tf.random_mormal([30, 3]), name='weight3')
    b3 = tf.Variable(tf.random_mormal([3]), name='bias3')
    hypothesis = tf.matmul(layer2, W3) + b3

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    feed_dict = {X:X, Y:Y}
    sess.run([cost, optimizer], feed_dict=feed_dict)

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('accuracy:', sess.run(accuracy, feed_dict={X:X, Y:Y}))
'''