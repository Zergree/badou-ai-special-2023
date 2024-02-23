import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#生成初始数据
x_data=np.linspace(-10,10,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.sin(x_data)+noise

#定义两个存放的地方
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#定义中间层
Weights_L1=tf.Variable(tf.random_normal([1,10]))    #随机一个weight
biases_L1=tf.Variable(tf.zeros([1,10]))  #偏置项
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
L1=tf.nn.tanh(Wx_plus_b_L1) #双曲正切函数

#定义输出层
Weights_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)
prediction=tf.nn.tanh(Wx_plus_b_L2)

# 定义损失函数
loss=tf.reduce_mean(tf.square(prediction-y))
# 定义反向传播算法  0.1是学习率
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #初始化
    sess.run(tf.global_variables_initializer())
    # 开始训练
    for i in range(1000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    prediction_final=sess.run(prediction,feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_final, 'r-', lw=5)  # 曲线是预测值
    plt.show()
