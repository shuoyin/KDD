# KDD
[Link to KDD CUP 2017](https://tianchi.aliyun.com/competition/information.htm?raceId=231597)<br />
我们选择Task 1: To estimate the average travel time from designated intersections to tollgates

## 输入输出
因为要根据前两个小时的数据预测后两个小时的平均通过时间,以20分钟为一个窗口, 我们对输入的数据也按20分钟时间窗口划分.
输入前两个小时的平均通过时间和天气特征,输出为一个6\*1维向量,表示后两个小时每20分钟里车辆的平均通过时间

## 模型选择
因为输入数据是时间上连续的序列,输出又是6维,所以我们选择LSTM处理序列数据.输入分为6个time step, 表示前两个小时每20分钟的
特征,输出分为6个time step,分别表示后两个小时每20分钟的平均通过时间.<br />
考虑到我们要根据输入的6个time step的全部信息来预测后6个time step, 我们采用了[Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)一文中提出的encoder-decoder模型<br />
因为天气数据在两个小时之内基本上是不变的,为了减少模型参数,输入LSTM的数据是当前time step的平均通过时间有关的特征,平均通过时间,道路的拥挤程度(根据平均通过时间划分等级)和周几.输入encoder-decoder的每个time step是这三维特征,根据这些信息预测出来的后两个小时的平均通过时间并没有考虑到当时的天气状况.
所以我们在decoder后边接了一个三层的神经网络,把天气特征连同预测的时间输入,输出调整之后的预测时间<br/>
下面是模型,用keras实现<br/>
```python
inputs = Input((seq_length,lstm_feature_dim))
l1 = LSTM(50,input_shape=(seq_length,lstm_feature_dim),return_sequences=False)(inputs)
l2=RepeatVector(6)(l1)
l3 = LSTM(50,return_sequences=True)(l2)
auxiliary_input = Input(shape=(6,auxiliary_feature_dim))
con = concatenate([l3, auxiliary_input])
d1 = TimeDistributed(Dense(100))(con)
d1 = TimeDistributed(Dense(1))(d1)
model = Model(input = [inputs, auxiliary_input], output = d1)
rms = optimizers.RMSprop(lr=0.005)  
model.compile(loss="mae", optimizer=rms)
```

## 数据预处理
训练数据是7月19号到10月17号全天的6条道路的每辆车的通过时间.<br />
我们把数据按道路分开,分别训练模型.对每一条道路,以20分钟为长度,统计20分钟区间内的平均通过时间,区间之间没有重叠.<br/>
有的时间段没有车辆通过,就先用NaN标记,然后统计其他日期里同一时段的平均通过时间,用这些记录的中位数代替<br />
有的时间段平均通过时间过大,比如有平均通过时间长达一千多分钟的时间段,对于这些明显的异常值,我们设定一个阈值,大于这个阈值的即为异常值,
用NaN标记,按缺失值对待.

## 参数调整
没有进行参数调整:sob::sob::sob:
