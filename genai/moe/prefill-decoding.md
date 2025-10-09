# 추론 최적화 개요 (Prefill과 Decoding에 따른 주요 기법 정리)

{% hint style="success" %}
이미 기본적인 LLM 추론 최적화에 대해 파악하고 계신 분들은 본 문서를 스킵해도 됩니다. MoE 모델 분산 서빙을 이해하기 위해서 필요한 선수 지식입니다.
{% endhint %}

트랜스포머 디코딩 기반 모델(주로 LLM/VLM)의 추론 과정은 본질적으로 두 개의 상이한 연산 특성을 가진 단계로 구성됩니다. Prefill 단계는 입력 프롬프트의 모든 토큰을 병렬로 처리하며 compute-bound 특성을 보이는 반면, decoding 단계는 토큰을 순차적으로 생성하면서 memory-bandwidth-bound (이하 memory-bound) 특성을 나타냅니다. Prefill은 행렬-행렬 연산으로 GPU의 계산 능력을 포화시키는 반면, decode는 행렬-벡터 연산으로 GPU 계산 유닛을 충분히 활용하지 못하고 메모리 대역폭에 의해 제한됩니다.

{% hint style="warning" %}
직관적인 이해를 위해 일부 용어는 번역하지 않고 원문 용어를 그대로 사용합니다. 물론 compute-bound를 연산 제한, memory-bandwidth-bound를 메모리 (대역폭) 제한이라 번역하기도 하지만 경험적으로 많은 엔터프라이즈 고객들이 이해하는 데 어려움을 겪었습니다.&#x20;
{% endhint %}

<figure><img src="../../.gitbook/assets/prefill-decode.png" alt=""><figcaption><p>LLM의 Prefill 단계와 Decode 단계</p></figcaption></figure>

## 1. 루프라인 차트/Prefill/Decode

***

### **1.1. 루프라인 차트 (Roofline chart)**

루프라인 차트<sup>Roofline chart</sup>는 HPC와 파운데이션 모델 최적화 분야에서 compute-bound와 memory-bound 성능 병목을 시각적으로 분석하기 위해 사용하는 성능 시각화 도구입니다. 이 차트를 활용하여 특정 연산이 compute-bound인지, 아니면 memory-bound인지를 직관적으로 판단합니다.

<figure><img src="../../.gitbook/assets/roofline-overview.png" alt=""><figcaption><p>루프라인 차트 (출처: <a href="https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html">https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html</a>)</p></figcaption></figure>

#### 차트 설명

* x축: 연산 집약도<sup>Arthmetic(또는 Operational) Intensity</sup> (OI)
  * 한 번의 메모리 접근(바이트)당 수행되는 부동소수점 연산(FLOP)의 수로 메모리 접근 1번당 얼마나 많은 계산을 하는가를 나타냅니다.
  * 단위: FLOPs / Byte
* y축: 성능<sup>Performance</sup>
  * 실제 측정된 또는 이론적 연산 속도
  * 단위: FLOPs/sec (예: TFLOPS)
* Memory Bandwidth Roof (왼쪽 기울어진 선)
  * 경사선 형태로 표시되며, 시스템의 메모리 대역폭(GB/s)에 의해 제한되는 성능 상한선입니다.
  * 이 선은 OI가 낮은 경우 즉, 메모리 접근이 많고 계산량이 적은 상황에서 병목이 되는 영역을 나타냅니다. 이 선보다 아래라면 메모리 접근량이 많고 연산량이 상대적으로 적은 memory-bound 상태입니다.
  * 파운데이션 모델 추론의 Decoding 단계에 해당하며, 성능 향상을 위해서는 메모리 재사용률 증가, 캐시 최적화, 데이터 압축 등이 필요합니다.
* Compute Roof (오른쪽 수평선)
  * 시스템의 최대 연산 성능(FLOPs/sec)으로 표시됩니다.
  * OI가 충분히 높아 메모리 병목을 넘어서면, 이제는 연산 유닛(FPU, Tensor Core 등)의 처리 능력이 한계가 됩니다. 이 선보다 아래라면 연산량이 상대적으로 많은 compute-bound 상태입니다.
  * 파운데이션 모델 추론의 Prefill 단계에 해당하며 성능 향상을 위해서는 벡터화, 병렬화, mixed precision, 커널 퓨전 등 연산 효율 개선이 핵심입니다.
* Ridge point
  * 두 선이 만나는 점을 ridge point(능선점)이라고 하며, 이 점이 메모리 대역폭 병목에서 연산 병목으로 전환되는 경계입니다.
* Achieved Value
  * 분석하려는 연산(예: GEMM, matmul, softmax 등)을 측정하여 해당 연산의 OI(x값)와 성능(FLOPs/sec, y값)을 구한 뒤 그래프 상에 점으로 표시합니다. 이 점과 Memory Bandwidth Roof/Compute Roof의 상대 위치로, 어느 병목이 주요 성능 제약인지 직관적으로 파악할 수 있습니다.

### **1.2. Prefill 단계: Compute-bound 구간**

LLM 추론 과정에서 prefill 단계는 전체 입력 프롬프트를 한 번에 처리하는 초기 구간입니다. 이때 모델은 입력 시퀀스의 모든 토큰에 대해 어텐션과 피드포워드<sup>feed-forward</sup> 연산을 동시에 수행하므로, 거대한 행렬곱(GEMM<sup>General Matrix Multiplication</sup>)과 텐서 연산이 집중적으로 발생합니다. GPU의 텐서 코어나 매트릭스 연산 유닛이 거의 포화 상태로 작동하며, 메모리 접근보다는 연산 수행 속도가 전체 처리량을 결정하는 compute-bound 특성을 보입니다.

성능을 향상시키려면 더 많은 연산을 같은 시간에 효율적으로 수행하여 루프라인 차트 수평선에 가깝게 끌어올리는 것으로, 커널 최적화와 연산 효율 증대가 핵심 전략입니다. 대표적인 기법은 큰 어텐션 행렬을 작은 타일로 나누어 빠른 SRAM에서 계산하는 연산 효율을 높이는 FlashAttention, 여러 연산을 하나의 GPU 커널로 통합해 오버헤드를 감소하는 커널 퓨전<sup>kernel fusion</sup>, 양자화<sup>quantization</sup>, FP16/BF16 혼합 정밀도<sup>mixed precision</sup> 등이 있습니다.

### **1.3. Decoding 단계: Memory-bound 구간**

Decoding 단계는 LLM이 응답을 생성할 때 수행되는 반복적인 토큰 생성 구간입니다. 이 단계에서는 매번 하나의 토큰만 처리하기 때문에 행렬곱 연산의 규모는 작지만, 각 토큰마다 과거 모든 토큰의 Key/Value 캐시(KV cache)를 다시 읽어와야 합니다. 즉, 연산량은 상대적으로 적지만 메모리 접근량이 매우 크기 때문에, GP의 연산 유닛이 아니라 메모리 대역폭이 성능을 제한하게 됩니다. 이 때문에 decoding은 전형적인 memory-bound 영역으로 분류됩니다.

성능 향상은 더 많은 연산을 하는 것이 아니라, 같은 대역폭으로 더 많은 데이터를 재사용하거나 더 적은 데이터를 읽는 것을 통해 이뤄집니다. 대표적인 기법은 KV 캐시 압축/양자화, 블록 기반 메모리 관리 기법인 PagedAttention, 중복된 프롬프트를 부분적으로 재사용하는 Prefix Caching, 작은 보조 모델이 여러 토큰을 예측하고, 큰 모델이 한 번에 검증함으로써 디코딩 단계를 줄이는 추측 디코딩<sup>speculative decoding</sup>이 있습니다.

#### 컴퓨트 바운드 vs 메모리 바운드 비교 요약

| 구분                    | Memory-bound           | Compute-bound            |
| --------------------- | ---------------------- | ------------------------ |
| 특징                    | 메모리 대역폭이 병목            | 연산 장치의 계산력이 병목           |
| Operational Intensity | 낮음 (메모리 접근이 많음)        | 높음 (연산이 집중적임)            |
| 루프라인 차트 상 위치          | 경사선(왼쪽) 아래             | 수평선(오른쪽) 아래              |
| 개선 방향                 | 데이터 재사용, 캐시 최적화, IO 감소 | 병렬화, 커널 퓨전, quantization |
| LLM 예시                | KV-cache 접근, 디코딩 단계    | 프리필 단계의 matmul, FFN 연산   |



## 2. Prefill/Decoding 단계 최적화

***

### 2.1. Prefill 단계 최적화: Compute-Bound 극복

#### 연산 특성과 병목 지점

Prefill 단계에서는 입력 시퀀스의 모든 토큰이 트랜스포머의 각 레이어를 동시에 통과하며, 이 과정에서 self-attention 메커니즘은 $$O(N²)$$ 복잡도의 행렬 곱셈을 수행합니다. 여기서 $$N$$은 시퀀스 길이를 의미하며, 이러한 연산은 본질적으로 계산 집약적입니다. GPU의 텐서 코어<sup>Tensor Core</sup>와 같은 고성능 연산 유닛을 최대한 활용하는 것이 핵심이지만, 시퀀스 길이가 증가함에 따라 메모리 요구량이 급격히 증가하는 문제가 발생합니다.

#### FlashAttention: IO-Aware 최적화

FlashAttention은 GPU의 메모리 계층 구조를 명시적으로 고려하여 HBM<sup>High Bandwidth Memory</sup>과 SRAM 간의 데이터 이동을 최소화하는 IO-aware 알고리즘입니다. 전통적인 attention 구현은 중간 결과인 어텐션 점수 행렬 $$QK^T$$ ($$N×N$$)을 HBM에 기록하고 다시 읽어오는 과정에서 상당한 오버헤드가 발생합니다. Softmax, dropout, masking과 같은 element-wise 연산들이 행렬 곱셈보다 더 많은 시간을 소비하는 memory-bound 특성을 보입니다.

FlashAttention은 이를 작은 타일<sup>tile</sup> 단위로 나누고, 온칩 SRAM(shared memory / register 수준) 내부에서 가능한 연산과 축적<sup>accumulation</sup>을 수행하여, 중간 결과를 HBM에 저장하지 않는 방식으로 메모리 접근을 최소화합니다.

구조적으로는 Q, K, V를 블록 단위로 분할해 작게 읽고 처리하며, softmax 계산에서도 online scaling(누적 최대값 갱신 방식 등)을 사용해 블록 경계를 넘나드는 정규화 문제를 안정적으로 처리합니다.  이 방식 덕분에 FlashAttention은 동일한 정확성을 유지하면서도, 종래 어텐션 구현 대비 2-4배 수준의 실제 속도 개선을 보고했으며, BERT-large (길이 512) 기준 15% 향상, GPT-2 (길이 1,000) 기준 3배, 장문 영역(long-range)에서도 2.4배 개선 효과를 보여 주었습니다.

하지만 FlashAttention V1은 GPU의 스레드 블록 할당 및 warp 간 커뮤니케이션, 공유 메모리 효율, 워프 점유율<sup>occupancy</sup> 측면에서 최적화 여지가 남아 있었습니다.

<figure><img src="../../.gitbook/assets/flash-attention.png" alt=""><figcaption><p>FlashAttention 개요 (출처: <a href="https://arxiv.org/pdf/2205.14135">https://arxiv.org/pdf/2205.14135</a>)</p></figcaption></figure>

#### FlashAttention-2

FlashAttention-2는 V1의 한계를 보완하기 위해 워크 분할<sup>work partitioning</sup> 전략을 개선하고, 커널 내부 병렬화를 세분화한 버전입니다.

구체적으로, V1에서는 단일 헤드나 타일 단위 작업이 하나의 스레드 블록 또는 warp 수준에서만 처리되는 경우가 있어서, GPU의 자원 활용률이 낮아지는 경우가 많았습니다. FlashAttention-2는 스레드 블록 간, warp 간 작업을 재분할하여 병렬도를 더 높이고, 특히 시퀀스 길이 방향(즉, $$N$$차원)에도 분할을 도입하여 단일 어텐션 head나 타일이 여러 블록에 걸쳐 병렬 처리될 수 있게 설계했습니다.

또한, non-matmul 연산 (예: softmax 재조정, scaling 과정 등)을 최소화하거나 커널 내부에서 효율적으로 배치해 텐서 코어 중심의 연산 흐름을 보조하도록 조정하였습니다.  이 결과, FlashAttention-2는 A100 등 GPU에서 이론 최고 성능의 50–73 % 수준까지 접근하는 효율을 보였고, 기존 FlashAttention 대비 약 2배 속도 개선을 입증했습니다.

그럼에도 불구하고, FlashAttention-2는 최신 아키텍처(예: NVIDIA Hopper 계열 GPU)에서 제공하는 비동기 명령 또는 메모리 이동 가속 기능을 충분히 활용하지 못해, 일부 GPU에서는 여전히 GEMM 수준의 효율에는 미치지 못하는 경우가 있었습니다.

#### FlashAttention-3

FlashAttention-3은 특히 Hopper 아키텍처 기반 GPU (예: H100, H800 등)를 타겟으로, V2가 놓친 하드웨어 특성을 적극 활용한 최적화를 추가한 커널로 3가지 핵심 기법이 도입되었습니다.

1. **비동기 오버랩(Asynchrony overlap)**: 텐서 코어 및 TMA(Tensor Memory Accelerator) 같은 하드웨어 유닛을 비동기적으로 동작시키면서 데이터 이동과 연산을 겹치게 스케줄링합니다. 이로써 메모리 이동과 연산이 병목 없이 중첩될 수 있게 설계되었습니다.
2. **Blockwise Matmul-Softmax 중첩**: 블록 수준 행렬곱<sup>matmul</sup>과 softmax/재스케일 연산을 중첩<sup>interleave</sup> 수행함으로써 유휴 시간이 적도록 커널 흐름을 최적화합니다.
3. **블록 양자화 (Block Quantization) 및 incoherent 처리**: FP8 같은 저정밀 연산을 활용하되, 양자화 오류를 제어하기 위한 변형 방식<sup>incoherent processing</sup>을 도입하여 낮은 수치 오류로도 성능을 유지할 수 있게 설계되었습니다.

이런 최적화 덕분에 FlashAttention-3은 FP16 기준으로 A100 대비 1.5–2배 빠른 성능을 보이며, H100에서는 약 740 TFLOPS (약 75% 활용률)까지 도달한 실험 결과를 보고합니다. FP8 모드에서는 1.2 PFLOPS 수준까지 확장 가능한 성능을 보이며, 동일 정밀도 대비 2.6배 낮은 수치 오차<sup>numerical error</sup>를 기록했다는 결과도 제시합니다.

#### Mixed Precision과 텐서 코어 활용

Prefill이 compute-bound라는 특성을 활용하면 정밀도를 낮추어 처리량을 증가시킬 수 있습니다. FP16 또는 BF16을 사용하면 FP32 대비 약 2배, INT8을 사용하면 4배의 이론적 성능 향상이 가능합니다. 특히 NVIDIA의 텐서 코어는 FP16/BF16 행렬 곱셈에 특화되어 있어, 이를 활용하면 일반 CUDA 코어 대비 약 8배의 처리량을 달성할 수 있습니다.

양자화<sup>Quantization</sup> 기법도 유사한 맥락에서 활용됩니다. Post-training quantization(PTQ)은 추가 학습 없이 가중치를 INT8 또는 INT4로 양자화하여 메모리 사용량과 데이터 이동량을 줄이면서, compute-bound 영역에서는 낮은 정밀도 연산의 높은 처리량을 활용합니다.

#### Batch 최적화: Continuous Batching 및 Dynamic SplitFuse

Prefill 단계의 병렬성을 극대화하기 위해 여러 요청의 prefill을 큰 배치로 묶어 처리하는 것이 효과적입니다. Prefill은 대규모 정적 워크로드를 선호하며, 여러 프롬프트를 배치로 처리하여 GPU 활용률을 극대화할 수 있습니다. 이는 행렬 차원을 증가시켜 GPU의 compute 유닛을 더 효율적으로 포화시킵니다.

그러나 서로 다른 길이의 프롬프트를 배치로 묶을 때 padding으로 인한 비효율이 발생할 수 있습니다. 이를 해결하기 위해 dynamic batching 또는 continuous batching 기법이 사용되며, 이는 유사한 길이의 요청들을 동적으로 그룹화하여 padding 오버헤드를 최소화합니다. 다만, continuous batching은 요청별 프롬프트 길이 차이로 인해 배치 내 연산량이 크게 불균형해지는 문제가 있습니다. 긴 프롬프트가 포함된 요청은 다른 요청들의 진행을 지연시키고, 배치 단위로 처리 크기가 달라지면서 GPU 활용 효율이 떨어집니다.&#x20;

DeepSpeed-FastGen에서 제안된 Dynamic SplitFuse는 단순히 요청을 합치는 수준이 아니라 프롬프트 자체를 청크 단위로 분할<sup>split</sup>하고, 이 청크들을 디코딩 연산과 유연하게 융합<sup>fuse</sup>하여 배치 크기를 일정하게 유지하도록 설계되었습니다. 즉, continuous batching이 “요청 단위의 동적 배치 구성”이라면, Dynamic SplitFuse는 “요청 내부 구조까지 동적으로 재조합”하는 보다 세밀한 접근입니다. 이로써 긴 프롬프트로 인한 지연 편차를 줄이면서도 GPU의 연산 파이프라인이 꾸준히 채워지도록 유지할 수 있습니다. 구체적으로 Dynamic SplitFuse는 다음 두 축의 동작을 결합합니다:

1. 긴 프롬프트는 그대로 한 번에 처리하는 대신 작은 청크 단위로 분할하여 여러 forward pass에 나눠 넣고, 최종 청크에서만 실제 토큰 생성을 수행합니다. 이렇게 하면 각 forward pass의 크기가 극히 커지지 않아 지연이나 불균형이 커지지 않습니다.&#x20;
2. 프롬프트가 짧아서 전체 배치 목표 크기(target token budget)에 못 미치는 경우, 프롬프트 일부를 “미리 생성 토큰 영역과 섞어서(compose)” 배치 규모를 채우도록 조정합니다. 즉, 프롬프트 + 생성 토큰을 하나의 forward pass로 묶어 크기를 맞추는 방식입니다.&#x20;

이 방식의 주요 이득은 forward pass 크기의 일관성을 유지하면서도 긴 프롬프트 처리에서 발생하는 latency 급증을 완화할 수 있다는 점입니다. 실제 실험에서는 이를 통해 vLLM 대비 최대 2.3배 유효 처리량 증가, 평균 2배 레이턴시 감소, 그리고 토큰 단위 P95 레이턴시에서 최대 3.7배 개선을 보고한 바 있습니다. &#x20;

<figure><img src="../../.gitbook/assets/continuous-batching.png" alt=""><figcaption></figcaption></figure>

#### Chunked Prefill과 Decode-Maximal Batching

Chunked prefill은 큰 prefill을 관리 가능한 작은 chunk로 분할하여 decode 요청과 함께 배치 처리할 수 있도록 합니다. 이는 prefill이 decode를 blocking하는 문제를 완화하며, GPU가 항상 유용한 작업을 수행하도록 보장합니다. [Sarathi 논문에서 제안된 decode-maximal batching](https://arxiv.org/abs/2308.16369)은 decode 요청을 우선적으로 스케줄링하고, 남은 계산 예산으로 chunked prefill을 처리하는 전략입니다.

이 기법은 prefill과 decode 간의 균형을 맞추면서도 전체 GPU 활용률을 향상시킵니다. 특히 실시간 서비스 환경에서 사용자가 기다리고 있는 decode 요청의 지연을 최소화하면서도, 새로 들어오는 긴 프롬프트를 효율적으로 처리할 수 있습니다.

### 2.2. Decoding 단계 최적화: Memory-Bound 극복

#### 연산 특성과 병목 지점

Decoding 단계는 auto-regressive하게 한 번에 하나의 토큰을 생성하며, 각 단계에서 모든 이전 토큰의 key와 value 정보에 접근해야 합니다. 이는 행렬-벡터 연산으로, GPU의 계산 능력을 충분히 활용하지 못합니다. 데이터(가중치, key, value, activation)가 메모리에서 GPU로 전송되는 속도가 latency를 지배하며, 실제 계산 속도는 부차적입니다.

작은 배치 크기에서 행렬 곱셈의 한 차원(배치 크기와 시퀀스의 토큰 수로 정의됨)이 작을 때, 연산은 메모리 대역폭에 제약을 받습니다. 이러한 상황에서 GPU의 peak compute 성능보다 peak memory bandwidth가 실제 처리량의 더 나은 예측 지표가 됩니다.

#### KV 캐싱

KV 캐싱은 decode 단계의 가장 기본적이면서도 필수적인 최적화입니다. Self-attention 계산에서 각 생성된 토큰이 모든 이전 토큰의 key와 value 텐서에 의존하므로, 이들을 GPU 메모리에 캐싱하여 재계산을 방지합니다. 일반적으로 모델의 각 레이어마다 별도의 KV 캐시가 유지되며, 이는 중복 계산을 대폭 감소시켜 decoding 과정을 가속화합니다.

하지만 KV 캐시는 메모리 소비의 주요 원인이기도 합니다. 긴 컨텍스트나 큰 배치 크기를 다룰 때 KV 캐시가 가중치보다 더 많은 메모리를 소비할 수 있습니다.

#### PagedAttention: 메모리 단편화 해결

KV 캐시는 종종 최대 입력(지원되는 시퀀스 길이)을 수용하기 위해 정적으로 "과잉 할당"됩니다. 입력 크기를 예측할 수 없기 때문에, 최대 시퀀스 길이가 2048이라면 실제 입력과 생성된 출력의 크기에 관계없이 2048 크기의 메모리가 예약됩니다. 이 공간은 연속적으로 할당되며, 대부분이 사용되지 않아 메모리 낭비와 단편화가 발생합니다.

PagedAttention은 운영체제의 가상 메모리 페이징<sup>paging</sup> 개념을 LLM 서빙에 적용한 기법입니다. KV 캐시를 관리 가능한 비연속적 메모리 블록으로 분할하여, 운영체제가 메모리 페이지를 관리하듯이 동적으로 할당하고 해제합니다. 각 시퀀스의 KV cache는 여러 블록(페이지)에 걸쳐 저장될 수 있으며, 블록 테이블을 통해 논리적 블록과 물리적 블록을 매핑합니다.

이 접근 방식은 메모리 활용률을 크게 향상시킵니다. 전통적인 연속 할당에서 약 20-40%의 메모리가 단편화로 낭비되는 반면, PagedAttention은 90% 이상의 메모리 활용률을 달성할 수 있습니다. 이는 더 큰 배치 크기를 가능하게 하여 전체 처리량을 향상시킵니다.

<figure><img src="../../.gitbook/assets/paged-attention.png" alt=""><figcaption><p>PagedAttention 개요 (출처: <a href="https://arxiv.org/abs/2309.06180">https://arxiv.org/abs/2309.06180</a>)</p></figcaption></figure>

#### Speculative Decoding (추측 디코딩): 병렬성 도입

추측 디코딩<sup>Speculative decoding</sup>은 decoding의 순차적 특성을 극복하기 위해 병렬성을 도입하는 혁신적인 패러다임입니다. LLM의 auto-regressive 디코딩은 순차적 계산을 요구하며, 각 단계가 이전 단계의 출력에 의존합니다. 이는 각 단계마다 전체 모델 파라미터를 HBM에서 가속기의 캐시로 이동해야 하는 병목을 만듭니다.

기본 아이디어는 작고 빠른 드래프트<sup>draft</sup> 모델을 사용하여 여러 토큰을 투기적으로 생성한 후, 대규모 타겟<sup>target</sup> 모델로 이들을 병렬로 검증하는 것입니다. 검증된 토큰들은 유지되고, 거부된 첫 토큰 이후는 폐기됩니다. 드래프트 모델이 토큰을 하나씩 제안하고, 타겟 모델은 단일 forward pass에서 이 토큰들을 검증하며, 올바른 토큰은 확인하고 잘못된 토큰은 수정합니다.

이 방법의 핵심 장점은 드래프트 모델의 예측이 틀려도 타겟 모델의 출력 분포는 변하지 않는다는 점입니다. 즉, 품질 저하 없이 속도만 향상됩니다. 실제로 적절한 드래프트 모델과 acceptance rate를 확보하면 수 배의 속도 향상이 가능합니다.

<figure><img src="../../.gitbook/assets/speculative-decoding.png" alt=""><figcaption></figcaption></figure>

#### Medusa: Self-Speculative Decoding

Medusa는 별도의 draft 모델을 유지해야 하는 복잡성을 제거하고, LLM에 여러 개의 디코딩 head를 추가하여 여러 후속 토큰을 병렬로 예측하는 효율적인 방법입니다. 원본 모델은 그대로 유지되고 새로운 head만 fine-tuning되는 parameter-efficient 방식입니다.

Medusa는 tree-based attention 메커니즘을 사용하여 여러 후보 continuation을 구성하고 각 decoding 단계에서 동시에 검증합니다. 각 head는 지정된 위치에 대해 여러 개의 top prediction을 생성하고, 이 예측들은 후보로 조립되어 tree-based attention 메커니즘을 통해 병렬로 처리됩니다.

Medusa-1은 기존 모델에 head만 추가하여 fine-tuning하며 generation 품질 저하 없이 2.2배 이상의 속도 향상을 달성합니다. Medusa-2는 backbone LLM과 함께 fine-tuning되어 더 나은 예측 정확도와 2.3-3.6배의 속도 향상을 제공하지만, backbone 모델의 능력을 보존하는 특별한 훈련 방법이 필요합니다.

### 2.3. Prefill과 Decode를 물리적으로 분리하는 Disaggregated Serving

최근 가장 주목받는 최적화 기법인 prefill 단계와 decode 단계를 아예 별도의 하드웨어 클러스터에서 처리합니다. 이렇게 하면 두 단계 간의 리소스 간섭<sup>interference</sup>을 줄이고, 각 단계에 최적화된 하드웨어/설정/병렬화 전략을 독립적으로 적용할 수 있게 됩니다.

<figure><img src="../../.gitbook/assets/disaggregated-serving.png" alt=""><figcaption><p>Disaggregated Serving (출처: <a href="https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models">NVIDIA 블로그</a>)</p></figcaption></figure>

Disaggregation의 핵심 컨셉은 각 단계의 고유한 병목을 독립적으로 최적화하는 것입니다. Prefill 서버는 프롬프트 배치를 병렬로 처리하고 KV cache를 생성합니다. Decode 서버는 캐싱된 KV state를 사용하여 효율적으로 토큰을 생성하며, 작은 배치 크기에서도 작동합니다. 이를 통해 더 높은 처리량(효율적인 배치와 병렬화), 더 낮은 지연(최적화된 메모리 접근과 전문 하드웨어), 그리고 파이프라인 병렬화(한 요청의 prefill을 다른 요청의 decode와 중첩)가 가능합니다.

### 2.4. Disaggregated Serving의 난제: **KV Transfer와 통신 오버헤드**

Disaggregated serving의 주요 난제는 prefill과 decode 인스턴스 간의 KV 캐시 전송입니다. KV 캐시 크기는 시퀀스 길이, 레이어 수, hidden dimension에 비례하여 상당히 클 수 있습니다. Disaggregated serving은 특히 latency가 중요한 애플리케이션에서 강력하지만, KV캐시의 효율적인 전송을 위해 RDMA<sup>Remote Direct Memory Access</sup> 기반 통신, 압축 기법, 그리고 전송 스케줄링 최적화가 필요합니다. 또한, vLLM과 AWS NeuronX Distributed(NxD)를 비롯한 다수 LLM 서빙 프레임워크가 실험적/제한적 지원 단계이기 때문에 프로덕션 적용을 고려한다면 충분한 테스트가 필요합니다.&#x20;

vLLM의 경우 KV connector 개념을 도입하여, prefill 쪽에서는 KV 블록을 외부 버퍼로 저장하고, decode 쪽에서는 이를 받아 해당 인스턴스의 KV 블록에 주입<sup>inject</sup>하는 흐름을 구현합니다. 이 전송은 비동기적으로 이루어지며, 모델의 주 연산과 겹치지 않도록 병렬화/스트리밍 방식으로 설계됩니다.

세부 동작은 다음과 같습니다:

*   **Prefill 측면:** 각 어텐 레이어의 계산이 끝나면, 해당 토큰의 K/V 블록을 CPU 메모리(또는 외부 버퍼)로 저장합니다.

    그 저장 작업은 메인 forward 연산과는 별도의 스레드/스트림을 사용하여 병렬로 수행됩니다.

    저장이 완료되면, 즉시 리모드 decode 인스턴스로 스트리밍 전송을 시작합니다.
* **Decode 측면:** 프록시를 통해 해당 요청의 prefill 인스턴스 주소를 알고, KV connector는 스트리밍 채널을 여러 개 열어 KV 캐시를 fetch(가져오기)합니다. 가져온 KV 블록은 임시 GPU 버퍼로 복사된 뒤, 로컬 vLLM의 KV 캐시 블록에 injection됩니다 (별도의 쓰레드/스트림에서). 이 주입이 끝나면 요청이 디코딩 스케줄러에 반환되고, 디코딩이 계속 진행됩니다.

NVIDIA Dynamo는 NIXL(NVIDIA InfiniBand eXtensions for LLMs)을 활용하여 GPU 간 직접 통신을 구현하며, zero-copy transfer로 latency를 최소화합니다. 외부 캐시 서버 역할을 하는 LMCache는 decoupled buffering을 통해 작은 page 크기를 유지하면서도 효율적인 KV 전송을 가능하게 합니다.



## 3. vLLM V1의 최적화 기법

***

vLLM V1은 2025년 1월에 알파 버전이 출시된 서빙 아키텍처로, 위에서 논의한 여러 최적화 기법을 통합하고 있습니다. vLLM 측에 따르면 V0 대비 1.7배의 스루풋<sup>throughput</sup> 향상을 달성하면서도 복잡도를 대폭 줄였습니다.

#### 통합 스케줄러

V1의 중앙집중식 스케줄러는 모든 요청에 대한 전역 최적화를 수행합니다. V0의 virtual engine 방식에서는 각 engine이 독립적으로 스케줄링하여 전역 최적화가 불가능했지만, V1은 단일 스케줄러가 전체 요청을 조망하여 최적의 배치를 구성합니다.

vLLM V1은 간단하면서도 유연한 중앙집중식 스케줄러를 도입합니다. 사용자로부터 주어진 프롬프트 토큰과 모델이 생성한 출력 토큰을 균일하게 취급함으로써 전통적인 “프리필(prefill)”과 “디코드(decode)” 단계의 구분을 제거합니다. 스케줄링 결정은 `{request_id: num_tokens}`와 같은 간단한 딕셔너리로 표현되며, 이는 각 단계에서 각 요청에 대해 처리할 토큰 수를 지정합니다.&#x20;

<figure><img src="../../.gitbook/assets/v1_scheduling.png" alt=""><figcaption><p>vLLM V1 스케줄링 (출처: <a href="https://blog.vllm.ai/2025/01/27/v1-alpha-release.html">https://blog.vllm.ai/2025/01/27/v1-alpha-release.html</a>)</p></figcaption></figure>

#### Prefill 최적화: Chunked Prefill

vLLM V1에서는 chunked prefill이 기본적으로 활성화되어 있습니다. 스케줄러는 decode 요청을 우선적으로 배치에 추가한 후, 남은 token budget(`max_num_batched_tokens`)으로 prefill 요청을 처리합니다. 마지막 prefill 요청이 budget을 초과하면 자동으로 chunk로 분할됩니다.

사용자는 `max_num_batched_tokens` 파라미터를 조정하여 워크로드 특성에 맞게 튜닝할 수 있습니다. 작은 값(512-1024)은 ITL을 우선시하고, 큰 값(4096-8192)은 TTFT<sup>Time To First Token</sup>를 우선시합니다. 이는 prefill과 decode 간의 균형을 유연하게 조정할 수 있게 합니다.

#### Decoding 최적화: Zero-Overhead Prefix Caching

vLLM V1의 가장 혁신적인 기능 중 하나는 zero-overhead prefix caching입니다. V0에서는 prefix를 식별하기 위해 매 요청마다 hash 계산이 필요했으며, 이는 TTFT의 10-20%를 차지했습니다. V1은 radix tree를 사용하여 토큰 시퀀스를 자동으로 추적하므로 hash 계산이 불필요합니다.

Request가 추가될 때 토큰 시퀀스를 tree에서 traverse하면서 매칭되는 토큰 수를 자동으로 계산합니다. 이미 cache된 prefix는 재사용되며, 이 과정의 오버헤드는 0.1ms 미만으로 측정됩니다. 반복적인 시스템 프롬프트를 사용하는 애플리케이션에서 특히 효과적입니다.

#### Continuous Batching

vLLM의 continuous batching은 고정된 batch size를 기다리지 않고 동적으로 요청을 그룹화합니다. 요청이 완료되면 즉시 배치에서 제거되고 새로운 요청이 추가됩니다. 이는 GPU가 항상 유용한 작업을 수행하도록 보장하며, 평균 대기 시간을 크게 줄입니다.

V1의 busy synchronous loop는 이를 더욱 효율화합니다. Scheduler는 매 iteration마다 실행 가능한 작업이 있는지 확인하고, 있다면 즉시 execution을 시작합니다. 이는 scheduling overhead를 최소화하며, V0 대비 약 90%의 overhead 감소를 달성합니다.

#### 여러 형태의 Speculative Decoding 지원

vLLM은 여러 형태의 speculative decoding을 지원합니다. 드래프트 모델 방식에서는 작은 모델을 사용하여 토큰을 제안하고 큰 모델로 검증합니다. N-gram matching 방식은 프롬프트 내의 n-gram을 매칭하여 후보 토큰을 생성합니다. Medusa 방식은 추가 decoding head를 사용하여 여러 토큰을 동시에 예측합니다.

사용자는 `speculative_model` 파라미터로 draft 모델을 지정하고, `num_speculative_tokens`로 예측할 토큰 수를 설정할 수 있습니다. 추측 디코딩은 특히 낮은 QPS(queries per second) 환경에서 상당한 성능 이점을 제공하며, 적절한 드래프트 모델 선택 시 최대 2.8배의 속도 향상이 가능합니다.

#### MoE 특화 전문가 병렬화

vLLM V1은 MoE<sup>Mixture-of-Experts</sup> 모델을 위한 전용 최적화를 제공하며 DeepEP와 PPLX 두 가지 dispatch/combine kernel을 지원합니다. DeepEP는 NVIDIA NVSHMEM을 기반으로 하며 멀티 노드 환경에서 우수한 성능을 보입니다. PPLX는 단일 노드 환경과 chunked prefill 시나리오에서 효과적입니다. Expert Placement with Load Balancing(EPLB)은 토큰 라우팅 편향으로 인한 불균형 로드를 자동으로 해결하며, 과도하게 활성화되는 전문가를 여러 GPU에 복제합니다.

#### 데이터 병렬화와 Disaggregated Prefill

vLLM V1은 데이터 병렬화를 통해 모델을 여러 replica로 복제하여 독립적인 요청 배치를 처리할 수 있습니다. 특히 MoE 모델에서는 어텐션 레이어를 DP로 복제하고 전문가 레이어는 EP로 분산시키는 하이브리드 접근법을 사용합니다. 이는 DeepSeek V2/V3/R1과 같은 Multi-head Latent Attention 모델에서 KV cache 중복을 방지하여 메모리 효율을 극대화합니다.

Disaggregated prefill은 실험적 기능으로 제공되며, LMCache와 NIXL 통합을 통해 prefill과 decode instance 간 효율적인 KV cache 전송을 가능하게 합니다. Prefill instance는 `kv_producer`로, decode instance는 `kv_consumer`로 설정되며, router가 요청을 적절히 분배합니다. 이는 TTFT와 ITL을 독립적으로 제어할 수 있게 하여 latency-critical 애플리케이션에 적합합니다.

#### 향상된 멀티모달 지원

vLLM V1은 텍스트와 이미지를 통합 처리하는 멀티모달 아키텍처를 제공합니다. 비전 인코더를 별도의 CUDA stream에서 실행하여 LLM forward pass와 오버랩시킬 수 있습니다. 이미지는 해시<sup>hash</sup>를 통해 캐싱되며, 동일한 이미지가 여러 요청에서 재사용될 때 인코딩을 생략합니다.

텍스트-이미지 interleaving이 완전히 지원되어 복잡한 멀티모달 시퀀스를 처리할 수 있습니다. Qwen-VL, InternVL, Phi-3-Vision과 같은 최신 비전-언어 모델뿐만 아니라 오디오 모델(Gemma Audio, Ultravox)와 비디오 모델도 네이티브하게 지원됩니다.

## References

#### Prefill 최적화

* Dao, T., et al. (2022). "[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.](https://arxiv.org/abs/2205.14135)" NeurIPS 2022.
* Dao, T. (2023). "[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.](https://arxiv.org/abs/2307.08691)" ICLR 2024.
* Dao, T., et al. (2024). "[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.](https://arxiv.org/abs/2407.08608)" NeurIPS 2024.

#### Decoding 최적화

* Kwon, W., et al. (2023). "[Efficient Memory Management for Large Language Model Serving with PagedAttention.](https://arxiv.org/abs/2309.06180)" SOSP.
* Shazeer, N. (2019). "[Fast Transformer Decoding: One Write-Head is All You Need.](https://arxiv.org/abs/1911.02150)" arXiv.
* Ye, Z., et al. (2025). "[FlashInfer: Accelerating Self-Attentions for LLM Serving.](https://www.arxiv.org/abs/2501.01005)" arXiv.

#### Speculative Decoding

* Leviathan, Y., et al. (2022). "[Fast Inference from Transformers via Speculative Decoding.](https://arxiv.org/abs/2211.17192)" ICML 2023.
* Cai, T., et al. (2024). "[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.](https://arxiv.org/abs/2401.10774)" ICML 2024.
* Fu, Y., et al. (2024). "[Break the Sequential Dependency of LLM Inference Using Lookahead Decoding.](https://arxiv.org/abs/2402.02057)" arXiv.

#### Disaggregated Serving

* Zhong, Y., et al. (2024). "[DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.](https://arxiv.org/abs/2401.09670)" OSDI 2024.
* Agrawal, A., et al. (2023). "[Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.](https://arxiv.org/abs/2308.16369)" arXiv.
* Ruhle, V., et al. (2025). "[POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference.](https://arxiv.org/abs/2410.18038)" ASPLOS 2025.
* vLLM Team. (2025). "[Disaggregated Prefilling.](https://docs.vllm.ai/en/stable/features/disagg_prefill.html)"
* vLLM Team. (2025). "[vLLM V1: A Major Architectural Upgrade.](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)" vLLM Blog (2025).
* Elmeleegy, A., et al. (2025). "[NVIDIA Dynamo, A Low-Latency Distributed Inference Framework for Scaling Reasoning AI Models](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models)." NVIDIA Blog (2025).&#x20;

