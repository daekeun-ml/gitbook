# QRNN(Quasi-Recurrent Neural Network)

## 1. Introductio&#x6E;**: RNN과 CNN의 한계**

* Can only process the input sequentially. → 병렬처리 비효율
* RNN은 이전 입력 데이터에 의존적임을 쉽게 알 수 있음.

$$
h_{t+1} = f(h_t, x_t), \\ \text{ where } x_t \text{ is the input at timestep } t, h_t \text{ is hidden state at }t
$$

* CNN은 모든 input에 동일한 weight를 적용(weight sharing)하므로 sequential dependency가 없기에 병렬처리가 원활
* 하지만, input sequence의 order 정보를 다룰 수 없음.
* RNN의 이점인 sequence order를 살리면서 CNN의 병렬처리 구조를 사용할 수 없을까?

## 2. Algorithm

![](https://housekdk.github.io/assets/imgs/2018-12-09-QRNN.png)

### **Convolution Layer**

* Convolution layer에서 세 개의 벡터를 계산; candidate vector, forget gate, output gate
* Given an input sequence of _n_-dim vectors $x\_1, x\_2, ... x\_T$, the convolution layer for the candidate vectors with _m_ filters produces a sequence of _T_ _m_-dimensional output vectors $z\_1, z\_2, ..., z\_T$.
*   수식이 직관적이지 않아 간단히 풀어쓰면 (_k_: filter width or filter size),

    $$z_t = \tanh(conv_{W_z}(x_t, ..., x_{t - k + 1})) \\ f_t = \sigma(conv_{W_f}(x_t, ..., x_{t - k + 1})) \\ o_t = \sigma(conv_{W_o}(x_t, ..., x_{t - k + 1}))$$
* 쉽게 말해 $&#x78;_{t-k+1}, \cdots , x_{t}$ 까지만 convolution을 수행함으로써 과거의 정보만 참조하며 미래의 정보는 참조하지 않음.

### **Pooling layer**

* 핵심: CNN의 convolution을 통해 이전 시점들(_t-k+1_)의 정보를 반영
* pooling layer에서 sequential processing을 최소화하고 많은 연산들을 convolution에 맡김으로써, 병렬처리에 용이
* LSTM과 상당히 유사한 수식이지만, 미묘하게 다름

#### _**f**_**-Pooling**

$$h_t = f_t \odot h_{t-1} + (1 - f_t) \odot z_t \;\; (\odot: \text{element-wise multiplication})$$

#### _**fo**_**-Pooling**

$$c_t = f_t \odot c_{t-1} + (1 - f_t) \odot z_t \\ h_t = o_t \odot c_t$$

* 자세히 보면 $c\_t$를 제외한 $z\_t, f\_t, o\_t$가 이전 시점에 의존적이지 않음

#### _**ifo**_**-Pooling**

$$c_t = f_t \odot c_{t-1} + i_t \odot z_t \\ h_t = o_t \odot c_t$$

## 3. **Variants**

### **Zone out**

* Dropout: 일부 activation을 bernoulli 확률로 0으로 만듦
* Stochastically chooses a new subset of channels to “zone out” at each timestep.
* 일부 activation을 이전 timestep의 activation으로 랜덤하게 대체
* 본 논문에서는 forget gate의 일부만 bernoulli 확률로 선택; _f_ gate의 subset을 1로 하거나, _1-&#x66;_&#xC5D0; dropout을 적용한다고 함.
* $$f_t^{new} = 1 - \text{dropout}(1- f_t), f_t = \sigma(conv_{W_f}(x_t, ..., x_{t - k + 1}))$$&#x20;
* [Zoneout 논문 (arXiv)](https://arxiv.org/abs/1606.01305)
* [Zoneout 구현 코드(GitHub)](https://github.com/teganmaharaj/zoneout)

### **Densely Connected Network(DenseNet)**

* Skip connection 사용
* 이전까지의 모든 layer를 concat하여 정보 보존 (ResNet은 add하는 방식)

## 4. **Encoder-Decoder Model**

![](https://housekdk.github.io/assets/imgs/2018-12-09-QRNN2.png)

### **Encoder**

* 각 layer마다 last hidden state $h\_{T}^{l}$를 계산하여 각 pooling layer에 추가

### **Decoder**

* Convolution 결과에 Encoder부에서 생성한 final encoder hidden state가 추가됨. (linearly projected copy of layer l’s last encoder state)

$$Z^{l} = \tanh(W_{z}^{l} \ast X^{l} + V_{z}^{l} \tilde{h_{T}^{l}}) \\ F^{l} = \sigma (W_{f}^{l} \ast X^{l} + V_{f}^{l} \tilde{h_{T}^{l}}) \\ O^{l} = \sigma (W_{o}^{l} \ast X^{l} + V_{0}^{l} \tilde{h_{T}^{l}})$$

$$l = l\text{-th layer}, h_{T}^{l} = \text{Encoder의 } l\text{-th layer의 마지막 hidden state}$$

## 5. **Experiments**

![](https://housekdk.github.io/assets/imgs/2018-12-09-QRNN3.png)

## References

* Paper
  * [https://arxiv.org/abs/1611.01576](https://arxiv.org/abs/1611.01576)
* Blog
  * [https://github.com/YBIGTA/DeepNLP-Study/wiki/QRNN](https://github.com/YBIGTA/DeepNLP-Study/wiki/QRNN)
* Code
  * [https://github.com/DingKe/nn\_playground/tree/master/qrnn](https://github.com/DingKe/nn_playground/tree/master/qrnn)
