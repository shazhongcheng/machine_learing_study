### 作业主要内容：

- [x] 我对pytorch和tensorflow的都进行了补全，然而由于对tensorflow2.0的不熟悉，我并没有将1.x的tensorflow的代码改成2.0的，同时也很明显是有问题的。
- [x] 解释一下 RNN ，LSTM，GRU模型:
1. RNN的公式结构为：
	$$h_{i}=\tanh(x_{i}*W_{xh}+h_{i-1}*W_{hh}+b_h)$$
    $$y_i=h_{i}*W_{hq}$$
2. LSTM比RNN要复杂，为了增强记忆，增加了记忆细胞C还有遗忘门F，此外还有输入门I和输出门O，公式结构为(不唯一)：
    $$F_{i}=sigmoid(dot(x_{i},W_{xf})+dot(h_{i-1},W_{hf}+b_f))$$
    $$I_i=sigmoid(dot(x_{i},W_{xi})+dot(h_{i-1},W_{hi}+b_i)$$
    $$O_i=sigmoid(dot(x_{i},W_{xo})+dot(h_{i-1},W_{ho}+b_o)$$
    $$C_{tilda}=\tanh(dot(x_{i},W_{xc})+dot(h_{i-1},W_{hc})+b_c)$$
    $$C_{i}=F_i*C_{i-1}+I_i*C_{tilda}$$
    $$H_{i}=O_i*tanh(C_{i})$$
    $$Y_i=dot(H_{i},W_hq)+b_q$$
3. 相较于LSTM，GRU在此基础上做了简化，将是三个门简化为两个，重置门R和更新门Z，同时去除了记忆细胞C
    $$Z_{i}=sigmoid(dot(x_{i},W_{xz})+dot(h_{i-1},W_{hz}+b_z))$$
    $$R_i=sigmoid(dot(x_{i},W_{xr})+dot(h_{i-1},W_{hr}+b_i)$$
    $$H_{tilda}=\tanh(dot(x_{i},W_{xh})+dot(R_i*Z_i,W_{hh})+b_h)$$
    $$H_{i}=Z_i*H_i+(1-Z_i)*H_{tilda}$$
    $$Y_i=dot(H_{i},W_hq)+b_q$$
- [x] 叙述一下 这个诗歌生成的过程。
    训练数据为[1,2,3,4,5]->[2,3,4,5,6]
    然后每一批次输入神经网络的数据格式为[seq,batch,embedding_dim]
    同时初始化h，用来记录时序信息
    训练好的模型，传入数据为[seq,1,embedding_dim]，初始化h,c为[layers,1,embedding_dim],传入模型，得到y和当前轮的h，c，接下来依次迭代。


