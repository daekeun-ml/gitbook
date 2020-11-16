# Transformer is All You Need

## 1. Background

* Encoder-Decoder 구조의 seq2seq한계를 CNN/RNN 없이 어텐션\(Attention\)만으로 극복
* **자동 회귀\(auto-regressive\) 모델**로, 한 번에 한 부분씩 예측하고 그 결과를 사용하여 다음에 수행할 작업 결정
* 장점
  * 데이터 전체의 시간적/공간적 관계에 대한 가정 없음
  * 병렬처리 가능
  * Long-term dependency에 강건
* NLP의 계보
  * 2001 - Neural language models
  * 2008 - Multi-task learning
  * 2013 - Word embeddings \(word embedding\)
  * 2014 - Sequence-to-sequence models
  * 2015 - Attention \(Attention Mechanism\)
  * 2015 - Memory-based networks
  * 2017 - Transformer
  * 2018 - Pre-trained language models \(2012년 AlexNet의 임팩트\)

## 2. Model Architecture

### Overview

* Encoder 블록과 Decoder 블록은 각 6개씩 존재
* 각 Encoder/각 Decoder들은 모두 동일한 구조를 가지고 있지만, weight를 공유하지는 않음
* Encoder
  * 2개의 sub-layer들로 구성: Self-Attention → Feed Forward

    ```jsx
      Stage1_out = Embedding512 + TokenPositionEncoding512 #w2v결과 + pos정보
      Stage2_out = layer_normalization(multihead_attention(Stage1_out) + Stage1_out)
      Stage3_out = layer_normalization(FFN(Stage2_out) + Stage2_out)
      ​
      out_enc = Stage3_out
    ```

  * 최대 시퀀스 길이\(예: 512토큰\)까지 입력을 받고 입력 시퀀스가 짧을 경우에는 padding으로 나머지 시퀀스를 채움
* Decoder
  * 3개의 sub-layer들로 구성 Self-Attention → Encoder-Decoder Attention → Feed Forward

    ```jsx
      Stage1_out = OutputEmbedding512 + TokenPositionEncoding512
      # i보다 작은 위치의 시퀀스 요소에 대해서만 어텐션 메커니즘이 동작할 수 있도록 마스킹한 어텐션
      Stage2_Mask = masked_multihead_attention(Stage1_out)
      Stage2_Norm1 = layer_normalization(Stage2_Mask) + Stage1_out
      Stage2_Multi = multihead_attention(Stage2_Norm1 + out_enc) + Stage2_Norm1
      Stage2_Norm2 = layer_normalization(Stage2_Multi) + Stage2_Multi
      ​
      Stage3_FNN = FNN(Stage2_Norm2)
      Stage3_Norm = layer_normalization(Stage3_FNN) + Stage2_Norm2
      ​
      out_dec = Stage3_Norm
    ```
* 각 sub-layer 사이에 Residual connection → Layer normalization으로 연결 \(Layer Normalization은 각 feature 대한 평균, 분산을 계산하고 각 example와 서로 독립적임\)
* [https://arxiv.org/pdf/1607.06450.pdf](https://arxiv.org/pdf/1607.06450.pdf) 참조 \(RNN과 트랜스포머에서 잘 작동한다고 알려짐\)

$$
y = LayerNorm(x + sublayer(x))
$$

## 3. Positional Encoding

* 위치 정보 반영; 사인\(sine\) 함수와 코사인\(cosine\)함수의 값을 임베딩 벡터에 더해줌으로써, 토큰의 상대적이거나 절대적인 위치 정보를 알려줌
* d차원 공간에서 문장의 의미와 문장에서의 위치의 유사성에 따라 토큰이 서로 더 가까워지게 함
* 가변 길이 시퀀스에 대해서 positional encoding 을 생성할 수 있기 때문에 scalability에서 큰 이점을 가짐 \(예를 들어, 이미 학습된 모델이 학습 데이터보다 더 긴 문장에 대해서 번역을 해야 할 때에도 positional encoding 생성 가능\)
* $$PE_{(pos,\ 2i)}=sin(pos/10000^{2i/d_{model}}) \\ PE_{(pos,\ 2i+1)}=cos(pos/10000^{2i/d_{model}})$$ 

## 4. Scaled Dot-product Attention

### Self-Attention

자기 자신\(Query\)을 잘 표현할 수 있는 \(key, value\) pair를 찾는 것

* 종래 seq2seq
  * Q = Query : _t-1_ 시점의 디코더 셀에서의 은닉 상태\(hidden state\)

    K = Keys : 모든 시점의 인코더 셀의 은닉 상태들

    V = Values : 모든 시점의 인코더 셀의 은닉 상태들
* Self-attention
  * Q : 입력 문장의 모든 토큰 벡터들

    K : 입력 문장의 모든 토큰 벡터들

    V : 입력 문장의 모든 토큰 벡터들
* 각 토큰 벡터\(임베딩+Positional Embedding\)로부터 Q, K, V 계산 \(가중치 행렬 Wq, Wk, Wv는 모델 파라메터\)
* Multi-Head Attention을 적용하기 때문에 Q, K, V 벡터 차원은 입력 벡터의 차원/Number of Heads임 \(논문에서는 512/8 = 64\)

### Scaled Dot-product Attention

* 각 Q 벡터는 모든 K 벡터에 대해서 어텐션 스코어를 구하고, 어텐션 분포\(attention distribution\)를 구한 뒤에 이를 사용하여 모든 V 벡터를 가중합\(weighted sum\)하여 어텐션 값 또는 컨텍스트 벡터\(context vector\)를 계산 → 이를 모든 Q 벡터에 대해서 반복
  * Q 벡터와 K 벡터의 내적\(dot-product\)으로 해당 query 벡터와 가장 유사한 key 벡터를 찾을 수 있고, 이를 softmax로 0~1 사이의 확률값으로 만들 수 있음.

    $$Attention(Q, K, V) = softmax({QK^T\over{\sqrt{d_k}}})V$$
* _**d\_k**_**의 제곱근으로 나누는 이유는 쿼리-키 내적 행렬의 차원이 늘어날 수록 softmax 함수에서 작은 기울기\(gradient\)를 가지기 때문에 이를 완화해 주기 위함.**
* 가중합으로 각 토큰 벡터에 대한 context 벡터 계산; 이 과정에서 관련이 있는 토큰의 비중이 높아지고 관련이 없는 토큰의 비중은 낮아짐.

### Decoder

* 미래 정보를 참조할 수 없으므로 현재 위치의 이전 위치들만 사용 가능
* 이를 위해 현재 time-step 이후의 위치들에 대해서 masking 처리 \(**softmax 적용 전에 -inf로 변경; 실제 구현은 -1e^9**; 왜냐하면 소프트맥스 함수에서 큰 음수 입력은 0에 가깝기 때문\)

### Code snippet

```python
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
```

## 5. Multi-Head Attention

* 모델이 다른 위치에 집중하는 능력 확장
* Attention layer가 여러 개의 “representation 공간”을 가지게 함 \(앙상블 효과\)
  * 선형 변환된 Q, K, V를 h개로 분리 → 위치가 상이한 각기 다른 표현 부분공간\(representation subspaces\) 블록들이 공동으로\(jointly\) 정보를 얻게 됨
  * h 개의 Attention Matrix가 생기면서 Q와 K간의 토큰들을 더 다양한 관점으로 볼 수 있음.
* Scaled Dot-Product 어텐션을 _h_번 계산해 이를 서로 concatenate \(논문의 경우 _h=8_\)
* h개의 행렬을 바로 Feed Forward Layer로 보낼 수 없기 때문에 또다른 가중치 행렬 Wo를 곱함

$$
\text{head}_i = \text{Attention}(Q W^Q_i, K W^K_i, V W^V_i) , i=1,\dots,h \\ \text{ where projections are parameter metrics }W^Q_i, W^K_i\in\mathbb{R}^{d_{model}\times d_k}, \\ W^V_i\in\mathbb{R}^{d_{model}\times d_v} \text{ for } d_k=d_v=d_{model}/h = 64
$$

$$
\text{MultiHead}(Q, K, V) = [\text{head}_1; \dots; \text{head}_h]W^O \\ \text{where } W^0\in\mathbb{R}^{d_{hd_v}\times d_{model}}
$$

### Code snippet

```python
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # 8
        self.d_model = d_model # 512

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
```

## 6. Position-wise Feed Forward Network

* **Position마다, 각 개별 단어 벡터마다 FFN\(Feed Forward Network\)이 적용됨**
* 두 개의 선형 변환\(linear transformation\); x에 선형 변환 적용 후, ReLU\(max\(0,z\)\)를 거쳐 다시 한번 선형 변환 적용;$$FFN(x)=max(0, xW_1+b_1)W_2+b_2$$
* 이때 각각의 position마다 같은 parameter W,b를 사용하지만, layer가 달라지면 다른 parameter 사용
* kernel size가 1이고 channel이 layer인 convolution을 두 번 수행한 결과와 동일 \(초기 구현 시 사용\)

  > While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1.

* kernel size를 3으로 조정 시 context 정보가 더 잘 표현된다고 하나, 현재 구글 공식 BERT 구현체는 convolution 연산 미적용
* Input/output dimension은 d\_model = 512이고, hidden layer dimension d\_ff = 2048임

## 7. Other Techniques

### Label Smoothing Regularization

* 원래 목표인 타겟 라벨을 맞추는 목적과 정답 라벨의 로짓\(logit\) 값이 학습 과정에서 과도하게 다른 logit 값보다 커지는 현상 방지
* [https://arxiv.org/abs/1512.00567](https://arxiv.org/abs/1512.00567) 참조할 것

### Optimizer

* Adam optimizer를 기반으로 warmup 기법 적용 \(학습률이 초기 step에서 가파르게 상승하다가 step 수를 만족하면 이후 천천히 하강\)

## References

* Paper
  * [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
* Blog
  * [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)
  * [https://pozalabs.github.io/transformer/](https://pozalabs.github.io/transformer/)
  * [https://wikidocs.net/31379](https://wikidocs.net/31379)
* Movie Clip
  * [https://www.youtube.com/watch?v=WsQLdu2JMgI](https://www.youtube.com/watch?v=WsQLdu2JMgI)
* Implementation
  * PyTorch: [http://nlp.seas.harvard.edu/2018/04/03/attention.html](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
  * TensorFlow: [https://www.tensorflow.org/tutorials/text/transformer](https://www.tensorflow.org/tutorials/text/transformer)
  * TensorFlow 예제 번역

    [transformer.html](Transformer%20is%20All%20You%20Need%20549c6cd814ad4cf18965e077543343db/transformer.html)

