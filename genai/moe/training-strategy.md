# 분산 훈련 전략

## 1. 실전 병렬 처리 조합 전략

***

### 1.1. 각 병렬화 기법 리마인더

* DP: 모델이 단일 GPU에 적재 가능하고 배치 확대 여지가 충분할 때 가장 단순하고 강력한 스케일-아웃 기법입니다.
* Sharded DP (ZeRO/FSDP): 모델이 단일 GPU 메모리에 안 들어갈 때 1순위로 검토합니다. 메모리 절감 극대화 + DP의 단순성을 유지합니다.
* TP: FFN/어텐션이 초대형이라 레이어 내부를 쪼개야 할 때 필요합니다. 다만 통신 오버헤드가 크므로 고품질 인터커넥트(예: NVLink, NVSwitch)가 필수입니다.&#x20;
* PP: 깊은 모델에서 메모리/계산을 스테이지로 분할할 때 유용하며 마이크로배치로 버블을 최소화해야 효과가 큽니다.
* EP: 희소 활성화로 FLOPs를 억제하며 모델 용량(파라미터 수) 을 크게 키우고 싶을 때 유용하며, all-to-all 통신 최적화가 관건입니다.

<table><thead><tr><th width="99.80859375">병렬화 기법</th><th>분할 대상</th><th>대표 통신</th><th>대표 라이브러리</th><th>강점</th><th>유의점</th></tr></thead><tbody><tr><td><strong>DP</strong></td><td>데이터 배치</td><td>Grad all-reduce</td><td>PyTorch DDP, NCCL</td><td>단순·안정, 확장 용이</td><td>모델은 1GPU 적재 필요</td></tr><tr><td><strong>Sharded DP</strong></td><td>파라미터/그래드/옵티마 샤딩</td><td>All-gather/Reduce-scatter</td><td>ZeRO(Stage 1-3), FSDP</td><td>메모리 절감 극대화</td><td>통신 증가·overlap 튜닝 중요</td></tr><tr><td><strong>TP</strong></td><td>레이어 내부 텐서</td><td>All-reduce/All-gather</td><td>Megatron-LM</td><td>초대형 레이어 수용</td><td>고성능 인터커넥트 의존</td></tr><tr><td><strong>PP</strong></td><td>레이어 구간(스테이지)</td><td>P2P/Collective</td><td>GPipe, DeepSpeed</td><td>메모리·계산 동시 분할</td><td>파이프라인 버블 관리</td></tr><tr><td><strong>EP</strong></td><td>전문가(FFN) 집합</td><td>All-to-all</td><td>DeepEP, Megatron Core</td><td>희소 활성+대용량</td><td>라우팅/불균형/통신 최적화 필수</td></tr></tbody></table>

### 1.2. 하이브리드 병렬 처리 (>=3D)

#### 개요

각 병렬 처리 기법은 단독으로는 한계가 있습니다.

* **DP만**: 모델이 GPU 메모리에 맞아야 함
* **TP만**: 노드 내로 제한, 통신 오버헤드 큼
* **PP만**: 파이프라인 버블<sup>Pipeline bubble</sup>로 효율성 저하
* **Sharded DP만**: 통신량이 많아 느린 네트워크에서 비효율적

따라서, 실무에서는 위 축들을 곱(product) 으로 조합해 확장합니다.

* 3D 병렬화(DP × TP × PP): Non-MoE 모델 훈련 베이스라인으로 2×2×2=8 GPU가 필요하며, 스케줄링/통신 최적화가 핵심입니다.&#x20;
* 4D/5D 병렬화: 3D 병렬화에 EP(또는 Megatron의 CP<sup>Context Parallel</sup>)까지 곱해 DP×TP×PP×EP(×CP)로 확장합니다.&#x20;

전문가 병렬화는 다른 공간 차원의 병렬화와 공존 가능하며, 서로 부족한 부분을 채워줍니다. 예를 들면 EP만으로 커버 안 되는 배치 크기 증가는 DP가, 단일 전문가의 크기 한계는 TP가, 모델 깊이 한계는 PP가, Conbtext 길이 한계는 CP가, 메모리 중복 문제는 Sharded DP가 각각 해결해줄 수 있습니다. 물론 복합 사용의 복잡도는 지수적으로 증가하여, 모든 것을 한꺼번에 쓰는 것은 구현상 무척 어렵습니다. DeepSpeed 팀은 MoE 시스템을 일컬어 다차원 병렬화의 교향곡(symphony of multidimensional parallelism)이라고 표현하였는데, 그만큼 섬세한 조율이 필요함을 시사합니다. 현재로서는 여러분이 모델 구조와 리소스 상황에 적합한 최적 조합을 휴리스틱하게 적용하는 수밖에 없지만, 앞으로 프레임워크들이 발전하면서 이러한 하이브리드 병렬 처리 전략도 더 사용하기 쉽게 추상화되고 자동 튜닝될 것으로 기대됩니다. EP를 비롯한 병렬화 기술의 꾸준한 혁신 덕분에 우리는 이전보다 훨씬 큰 모델을 빠르고 경제적으로 다룰 수 있게 되었습니다.

예컨대 Megatron-Core는 [0.9 버전에서 468 TFLOPS(Mixtral 8x7B)를 달성](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html)하며, Expert Tensor Parallelism (ETP) 지원으로 전문가 내부도 텐서 병렬화가 가능합니다. Shared Expert 기능은 dense 부분과 sparse 부분의 조합을 최적화하고, Data Parallelism + Tensor Parallelism + Pipeline Parallelism + Expert Parallelism + Context Parallelism의 5차원 병렬처리를 동시에 지원합니다.

2025년 현재 DeepSeek-V3, Mixtral 8x22B 등의 실용화된 MoE 모델은 기존 Dense 모델 대비 수 배의 성능 향상을 달성하고 있으며, 1조 파라미터 이상의 초대규모 모델에서도 25ms 이하의 추론 지연시간을 달성했습니다.

### 1.3. 병렬 조합 베이스라인

최적의 병렬 조합은 모델 아키텍처, 학습된 시퀀스 길이 및 하드웨어 플랫폼에 따라 달라지지만, 아래의 베이스라인 가이드를 따르는 것을 권장합니다.

#### 사전 체크리스트

{% hint style="success" %}
공통 가정

* GPU: A100 80GB/H100 80GB, BF16 훈련, NVLink(노드 내) + IB or EFA(노드 간)
* MoE 라우팅: Top-1/2, capacity factor 1.25 (안정/과밀 방지 밸런스)
* 시퀀스 길이: 4k 토큰, activation checkpointing 사용
* 통신 라이브러리: NCCL(+ RDMA), EP all-to-all 최적화(계층/2D 또는 통신-연산 overlap)
{% endhint %}

* **메모리 제약**: 우선 Sharded DP(ZeRO/FSDP) 로 1차 조정 후, 남는 병목을 PP/TP로 해소하는 접근을 권장합니다.&#x20;
* **인터커넥트**: TP/EP는 통신 집약적이므로, 가능하면 노드 내(예: NVLink) 에 배치하고, 노드 간 경로는 PP/DP/샤딩 위주로 구성합니다. EP 크기는 {4,8}로 실험하세요. P6-B200에서는 TP/EP 비율을 더 공격적으로 설정할 수 있습니다.
* **마이크로배치/스케줄**: PP는 마이크로배치 수와 1F1B/인터리브드 스케줄 최적화가 성능을 좌우합니다.&#x20;
* **EP 라우팅**: 토큰 불균형을 완화하는 계층/2D all-to-all(예: Tutel)이나 통신-연산 overlap을 적극 활용하세요.
* **오프로딩**: NVMe/CPU 오프로딩(ZeRO-Infinity 등)으로 메모리 벽을 넘을 수 있지만, 오프로딩은 최후의 수단입니다. 오프로딩 적용 시에는 I/O 병목을 모니터링해야 합니다.&#x20;

#### 상황별 우선순위

* **NVLink 있고 빠른 네트워크:** TP (노드 내) → PP (노드 간) → DP (전체)
* **느린 네트워크:** PP (노드 간) → TP (노드 내, 가급적 작게) → DP (전체)
* **메모리가 부족할 때:** ZeRO-2 활성화 → TP size 증가 → PP size 증가 → ZeRO-3 + Offload (최후)
* **처리량을 높이고 싶을 때:** Mixed precision (FP8/BF16) → DP size 증가 → Micro-batch 수 증가 → Overlapping 최적화

#### 파라미터 설정

**TP 크기를 가능한 작게 유지하세요.**

* 대형 언어 모델의 경우 OOM<sup>Out-of-Memory</sup>을 방지하기 위해 TP가 종종 필요하지만, 이는 통신 오버헤드를 발생시키고 성능을 저하시킬 수 있습니다.
* 분산 옵티마이저<sup>distributed optimizer</sup>를 사용할 때 마스터 가중치<sup>master weights</sup>와 옵티마이저 상태<sup>optimizer states</sup>는 모든 DP 랭크에 걸쳐 샤딩<sup>sharding</sup>되기에 약간의 통신 오버헤드만 발생합니다. 따라서 학습 중에 여유 GPU 메모리가 많을 경우 TP 크기를 줄이고 DP 크기를 늘리는 것이 좋습니다.

**EPxTP 통신이 노드 내 (예: NVLink 도메인 내)에서 이루어지는 것이 좋습니다.**

* 소형 MoE 모델의 경우 EP8 x TP1이 EP4 x TP2보다 더 좋습니다. MoE 레이어의 계산 그래프가 단순해져 잠재적인 통신-계산 중첩<sup>comm-computation overlapping</sup>을 수행하기에 더 편리하기 때문입니다. 30–70B 모델은 32–64 GPU로도 충분히 빠르게 돌릴 수 있습니다.
* 모델이 너무 커서 다중 노드에 걸쳐 확장해야 하는 경우에는 TP와 EP보다 PP를 먼저 고려하세요.&#x20;

**모델을 더 확장하려면 PP를 사용하세요.**

* Megatron-LM의 경우 `pp_size >= 2`일 때 `num_layers_per_virtual_pipeline_stage`를 설정하여 가상 파이프라인 병렬화(VPP<sup>Virtual Pipeline Parallelism</sup>)를 활성화하면 PP 버블을 줄일 수 있습니다.
* VPP\_size 튜닝: `vpp_size`의 합법적인 값은 모두 `num_layers/pp_size`의 공약수들입니다. 예컨대 `num_layers=24`, `pp_size=4`이면 vpp\_size는 `{1, 2, 3, 6}` 중에서 선택할 수 있습니다. `vpp_size`가 클수록 파이프라인 버블은 줄어들지만 각 PP 스테이지 간의 P2P 통신이 증가하므로, 휴리스틱하게 중간 값을 베이스라인으로 시작하세요.&#x20;
  * `VPP_size = num_layers / PP_size / num_layers_per_virtual_pipeline_stage`&#x20;

### 1.4. AWS 분산 훈련 베이스라인

#### AWS AI 인프라 설정

**NVLink**

* `nvidia-smi nvlink -s` 는 각 GPU의 모든 NVLink 상태를 확인하는 명령입니다. 아래는 p4de.24xlarge 예시입니다.

```sh
ubuntu@ip-172-31-35-99:~$ nvidia-smi nvlink -s
GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-370ec676-e407-3115-836a-8ebcb3c4f62a)
  Link 0: 25 GB/s
  Link 1: 25 GB/s
  Link 2: 25 GB/s
  Link 3: 25 GB/s
  Link 4: 25 GB/s
  Link 5: 25 GB/s
  Link 6: 25 GB/s
  Link 7: 25 GB/s
  Link 8: 25 GB/s
  Link 9: 25 GB/s
  Link 10: 25 GB/s
  Link 11: 25 GB/s
```

* `sudo dcgmi diag -r 2 -p pcie.gpu_nvlinks_expected_up=<# NVLinks>` 로 추가 검증합니다.

| 인스턴스               | GPU       | NVLink 수 | NVLink 세대 |
| ------------------ | --------- | -------- | --------- |
| p4de.24xlarge      | A100 80GB | 12       | 3세대       |
| p5.48xlarge        | H100      | 18       | 4세대       |
| p5en.48xlarge      | H200      | 18       | 4세대       |
| p6-b200.48xlarge   | B200      | 18       | 5세대       |
| p6e-gb200.36xlarge | B200      | 18       | 5세대       |

```sh
ubuntu@ip-172-31-35-99:~$ dcgmi diag -r 2 -p pcie.gpu_nvlinks_expected_up=12
Successfully ran diagnostic for group.
+---------------------------+------------------------------------------------+
| Diagnostic                | Result                                         |
+===========================+================================================+
|-----  Metadata  ----------+------------------------------------------------|
| DCGM Version              | 3.3.3                                          |
| Driver Version Detected   | 535.104.12                                     |
| GPU Device IDs Detected   | 20b2,20b2,20b2,20b2,20b2,20b2,20b2,20b2        |
|-----  Deployment  --------+------------------------------------------------|
| Denylist                  | Pass                                           |
| NVML Library              | Pass                                           |
| CUDA Main Library         | Pass                                           |
| Permissions and OS Blocks | Pass                                           |
| Persistence Mode          | Pass                                           |
| Environment Variables     | Pass                                           |
| Page Retirement/Row Remap | Pass                                           |
| Graphics Processes        | Pass                                           |
| Inforom                   | Pass                                           |
+-----  Integration  -------+------------------------------------------------+
| PCIe                      | Pass - All                                     |
+-----  Hardware  ----------+------------------------------------------------+
| GPU Memory                | Pass - All                                     |
+-----  Stress  ------------+------------------------------------------------+
+---------------------------+------------------------------------------------+
```

**EFA 체크**

* `fi_info -p efa` 로 EFA provider가 로드되는지 노드별로 점검합니다.
* NCCL 초기화 로그에 libfabric/EFA/GDRDMA 사용 표기가 찍히는지 확인하고 “EFA/libfabric path used” 같은 문구가 보이는지 점검합니다.

**Slurm**

* SageMaker HyperPod의 Slurm은 PMIx/PMI(Process Management Interface)를 통한 런치가 표준입니다. `--mpi=pmix` 조합을 사용하고 CPU 바인딩은 `--cpu-bind=cores` 를 권장합니다. (참조: [https://slurm.schedmd.com/mpi\_guide.html](https://slurm.schedmd.com/mpi_guide.html), [https://slurm.schedmd.com/mc\_support.html](https://slurm.schedmd.com/mc_support.html))
* HyperPod Slurm을 사용한다면, 2025년 9월 신규 런칭된 기능인 [헬스 모니터링 에이전트](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-resiliency-health-monitoring-agent.html)를 사용 가능합니다. 신규 Slurm 클러스터에서 자동으로 활성화되며, AMI 업그레이드를 통해 기존 클러스터에 추가할 수 있습니다.&#x20;

**버전 확인**

* CUDA / NVIDIA Driver / NCCL / PyTorch / libfabric / aws-ofi-nccl 버전을 확합니다. 특히 NCCL은 환경변수 기반 튜닝 폭이 넓으므로 **해당 버전의** NCCL 문서(Env Vars)를 꼭 확인하세요. 주요 변수(`NCCL_CROSS_NIC, NCCL_{MIN,MAX}_NCHANNELS` 등) 의미가 버전에 따라 미세하게 달라집니다.

**EFA 주요 환경 변수 (출처:** [**AWS EFA Cheatsheet**](https://app.gitbook.com/u/A81uOOpezEhPlFs0xy0rsEctTSP2)**)**

<table><thead><tr><th width="270.4765625">설정</th><th>설명</th></tr></thead><tbody><tr><td><code>NCCL_DEBUG=info</code></td><td>NCCL에서 디버그 정보를 얻기 위해 설정합니다. 이를 통해 NCCL이 EFA를 사용하고 있는지와 어떤 버전을 사용하고 있는지 확인할 수 있습니다. 다만, 많은 디버그 정보를 출력하므로 NCCL 문제가 의심되지 않는 한 끄는 것을 권장합니다. 자세한 내용은 <a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug">NCCL_DEBUG</a> 를 참조하세요.</td></tr><tr><td><code>FI_EFA_USE_HUGE_PAGE=0</code></td><td><code>os.fork()</code>가 <code>OSError: Cannot allocate memory</code>를 발생시킬 때 0으로 설정합니다. 일반적으로 다중 프로세스 PyTorch 데이터 로더에서 발생합니다. 거대 페이지를 비활성화하면 약간의 성능 저하가 있지만, 운영 체제의 거대 페이지 부족으로 인한 fork 실패를 방지하는 데 필요합니다.</td></tr><tr><td><code>FI_EFA_FORK_SAFE=1</code></td><td>kernel>=5.15에서는 필요하지 않습니다. 설정해도 효과는 없지만 문제없습니다. [<a href="https://github.com/ofiwg/libfabric/pull/9112">참조</a>]</td></tr><tr><td><code>FI_EFA_USE_DEVICE_RDMA=1</code></td><td><code>libfabric>=1.18.0</code> 및 <code>aws-ofi-nccl>=1.7.0</code>에서는 설정할 필요가 없습니다.</td></tr><tr><td><code>FI_EFA_SET_CUDA_SYNC_MEMOPS=0</code></td><td><code>efa-installer&#x3C;1.29.1</code> 및 <code>nccl>=2.19.0</code>에서 NCCL 오류 <code>register_rail_mr_buffer:617 NCCL WARN NET/OFI Unable to register memory (type = 2) for device 4. RC: -22, Error: Invalid argument</code>를 방지하기 위해 설정합니다.</td></tr><tr><td><code>FI_EFA_ENABLE_SHM_TRANSFER=1</code></td><td>필요하지 않습니다. 실제로는 <code>no-op</code>이며, 기본값이 이미 SHMEM을 활성화합니다.</td></tr><tr><td><code>FI_PROVIDER=efa</code></td><td><code>aws-ofi-nccl&#x3C;=1.5.0</code> 및 p4/p5 인스턴스에서 사용합니다.</td></tr><tr><td><code>NCCL_PROTO=simple</code></td><td><code>aws-ofi-nccl&#x3C;=1.5.0</code> 및 p4/p5 인스턴스에서 사용합니다.</td></tr><tr><td><code>NCCL_SOCKET_NTHREADS</code></td><td>EFA에는 적용되지 않습니다.</td></tr><tr><td><code>NCCL_SOCKET_IFNAME</code></td><td><code>p5.48xlarge</code>와 <code>p4d(e).24xlarge</code> 모두를 포함하려면 <code>en</code>으로 설정합니다. 다른 인스턴스의 경우 <code>ifconfig</code>를 확인하여 활성 네트워크 인터페이스를 확인하세요.</td></tr><tr><td><code>NCCL_NSOCKS_PERTHREAD</code></td><td>EFA에는 적용되지 않습니다.</td></tr><tr><td><code>NCCL_MIN_CHANNELS=xxx</code></td><td>기본값을 사용하도록 설정하지 않는 것을 권장합니다. 예를 들어, p4d/p4de에서 채널 수는 8이어야 하며, 이는 4-NIC 플랫폼의 최소값입니다. 감소 메시지는 작업의 GPU 수로 나누어진 다음 채널 수로 나누어지므로, 필요 이상의 채널을 가지면 더 작은 메시지가 발생하여 EFA가 데이터 부족 상태가 됩니다.</td></tr><tr><td><code>NCCL_BUFFSIZE=xxx</code></td><td>기본값을 사용하도록 설정하지 않는 것을 권장합니다.</td></tr><tr><td><code>RDMAV_FORK_SAFE=1</code></td><td>사용하지 마세요. 이는 RDMA-core 환경 변수입니다. <code>FI_EFA_FORK_SAFE</code>를 선호하세요(Linux 커널 버전에 따라 여전히 의미가 있는 경우). 둘이 비슷해 보이지만 실제로는 매우 다르게 동작하며, 특히 새로운 커널에서는 <code>RDMAV_FORK_SAFE=1</code>이 문제를 일으킬 수 있습니다.</td></tr><tr><td><code>NCCL_SHM_USE_CUDA_MEMCPY=1</code></td><td>g6/g5에서 NCCL을 실행할 때 설정합니다. 기본 memcpy에 비해 2배의 성능을 제공합니다.</td></tr><tr><td><code>RDMAV_*</code></td><td>사용하지 마세요.</td></tr><tr><td>NCCL 버전</td><td>안정적인 릴리스 중 하나를 권장합니다.</td></tr></tbody></table>

```bash
export FI_EFA_USE_HUGE_PAGE=0   # 기본 세팅

# libfabric/EFA 선택 
# export FI_PROVIDER=efa

# GPU↔EFA 디바이스 RDMA(가능 시) 사용 (
# export FI_EFA_USE_DEVICE_RDMA=1

# NCCL가 aws-ofi-nccl을 경유하도록 LD_LIBRARY_PATH 확인
# (플러그인은 자동 로드; 최신 AMI/빌드 기준)

export NCCL_DEBUG=INFO         # 디버그 시 사용
export NCCL_IB_GID_INDEX=3     # VPC/서브넷에 따라 조정
export NCCL_NET_GDR_LEVEL=PHB  # 안정성 이슈가 보이면 PHB → PXB → PIX로 단계적으로 보수화

# 알고리즘/프로토콜은 기본 Auto 권장. 병목 시 실험:
# export NCCL_ALGO=Tree|Ring
# export NCCL_PROTO=Simple|LL|LL128
```

* `NCCL_NET_GDR_LEVEL` (formerly NCCL\_IB\_GDR\_LEVEL): NCCL에서 GPUDirect RDMA(GDR)를 허용할 토폴로지 거리의 최대 수준을 제어하는 환경 변수입니다. 즉, 네트워크 전송 시 GPU 메모리를 직접 NIC(Network Interface Card)에 매핑해서 CPU 메모리 복사를 거치지 않고 통신할 수 있는 GDR 최적화 정도를 지정하는 옵션입니다. 보통 자동 최적화되지만 명시적으로 기재하는 것이 좋습니다.
  * `LOC` (Local) : GDR 항상 비활성화
  * `PIX` (PCIe Switch): GPU와 NIC가 같은 PCIe 스위치에 연결되어 있을 때만 GDR 사용. 가장 일반적이고 안전한 설정 중 하나
  * `PXB` (PCIe Bridge / multiple switches) : GPU와 NIC가 서로 다른 PCI 스위치에 걸려 있더라도, PCIe 브리지를 통해 연결되어 있으면 GDR 사용
  * `PHB` (PCI Host Bridge / NUMA node) : 같은 NUMA 노드(같은 CPU 소켓)면 GDR (트래픽은 CPU 루트를 통과 가능)
  * `SYS` (System-wide) : 소켓 간 링크(UPI/QPI)까지 넘어서도 항상 GDR 사용. GPU와 NIC가 시스템 내 어디에 있든 GPUDirect RDMA 활성화. 가장 공격적이고 호환성 리스크가 있음.
* 토폴로지 확인: `nvidia-smi topo -m`에서 GPU↔NIC 경로가 PIX/PXB/PHB인지 확인하고 그에 맞춰 GDR 레벨을 선택합니다.



## 2. 실전 체크리스트

***

{% hint style="success" %}
모델 크기에 따른 메모리 요구량을 GPU 메모리로 나누어서 몇 장의 GPU가 최소로 요구되는지 먼저 확인하세요.&#x20;

훈련 메모리 = P × B × (1 + gradient + optimizer states) = P × B × (1 + 1 + 4-8) \[Adam optimizer 기준] = P × B × 6-10

* FP16 Adam 기준
  * 파라미터: P × 2 bytes
  * Gradient: P × 2 bytes
  * Optimizer states: P × 8 bytes (FP32 copy + momentum + variance)
  * Total: P × 12 bytes
* 70B 모델 예시
  * 70B × 12 bytes = 840GB 최소 GPU 수 = 840GB / 80GB (A100 80GB 기준): 최소 11 GPU&#x20;
  * 실제로는 activation 등 추가 메모리 필요하기에 16-32 GPU 권장
{% endhint %}

### 2.1. 시작 전

**사전 점검**

* EFA/InfiniBand 같은 고속 네트워크가 제대로 구성되어 있는가?
* NCCL 버전이 최신인가?
* 파라미터 수와 메모리 요구량
* GPU 메모리가 충분한가? (최소 요구량 × 1.5배)

**병렬 처리 설정**

* TP, PP, DP 크기 결정 (TP × PP × DP = 총 GPU 수)
* MoE 모델인 경우 EP 크기 설정
* ZeRO stage 선택 (1/2/3)
* Micro-batch size 설정 (파이프라인 버블 고려)

### 2.2. 학습 중 모니터링

**성능 지표**

* GPU 메모리 사용률 (80-95%가 이상적)
* GPU 활용률 (70%+ 목표)
* TFLOPS per GPU 측정 (높을수록 좋지만 베이스라인은 처음에 낮게 가져가고 점차 올리는 것을 권장)
* 초당 처리 샘플 수 (samples/sec)

**통신 체크**

* All-reduce 지연시간 모니터링
* All-to-All 통신 시간 (MoE의 경우)
* Pipeline bubble ratio (10% 이하 목표)
* 통신/연산 오버랩 효율성

**안정성**

* Loss curve가 안정적인가?
* Gradient norm이 폭발하지 않는가?
* Out of memory 에러 없는가?
* Checkpoint 저장이 정상 동작하는가?

### 2.3. 최적화 체크포인트

**메모리 최적화**

* Activation checkpointing 활성화 여부 재검토
* ZeRO stage 상향 조정 가능성
* Offloading (CPU/NVMe) 필요성
* Gradient accumulation으로 배치 크기 조정

**통신 최적화**

* Overlap 전략 활성화 (compute/communication)
* Bucket size 튜닝 (all-reduce)
* Hierarchical communication 전략 (MoE)

**연산 최적화:**

* Flash Attention 활용
* Fused kernels 활성화
* Mixed precision 최적화
* Compiler 최적화 (CUDA graphs, TorchScript)

### 2.4 MoE 전용 체크리스트

**라우팅 모니터링**

* Expert utilization 균형 (20% 이내 편차)
* Load balancing loss 값 추적
* Token drop rate (10% 이하 목표)
* Capacity factor 조정 필요성

**통신 최적화**

* All-to-All vs. AllGather/ReduceScatter 비교
* Expert-wise batching 효율
* Hierarchical All-to-All (multi-node)

**메모리 관리**

* Expert parameter sharding 전략
* Shared expert 메모리 사용량

### 2.5 디버깅 체크리스트

**OOM (Out of Memory)**

* Micro-batch size 줄이기
* Activation checkpointing 활성화
* ZeRO stage 올리기
* TP/PP size 증가

**느린 훈련 속도**

* Pipeline bubble 확인 (micro-batch 수 증가)
* 통신 병목 확인 (네트워크 대역폭)
* Uneven layer distribution 확인 (PP)
* Expert load imbalance 확인 (MoE)

**수렴 문제**

* Learning rate 조정
* Gradient clipping 설정
* Warmup steps 증가
* Loss spike 시 checkpoint 복구

**통신 에러**

* NCCL 버전 확인
* Network timeout 설정 증가
* Process group 초기화 순서 확인
* Firewall/네트워크 설정 확인

### 2.6. 모델 규모별 추천 구성

{% hint style="success" %}
실전에서는 작게 시작해서 점진적으로 확장하는 것이 좋습니다. 7B 모델로 파이프라인을 검증하고, 문제를 해결한 후에 30B, 70B, 175B로 확장하는 것이 좋습니다.
{% endhint %}

#### Dense 모델

* **<= 7B 모델 추천 구성 (8 GPU / 1 노드)**
  * TP = 1 PP = 1 DP = 8 ZeRO Stage = 1 or 2
* **<= 13B 모델 추천 구성 (16 GPU / 2 노드):**
  * TP = 1, PP = 2, DP = 8, ZeRO Stage = 2, 또는 TP = 2, PP = 1, DP = 8, ZeRO Stage = 2
* **30B-70B 모델 추천 구성 (64 GPU / 8 노드):**
  * **옵션 1: TP 중심:** TP = 8 (노드 내) PP = 2 (노드 간) DP = 4 ZeRO Stage = 1
  * **옵션 2: PP 중심 (느린 네트워크):** TP = 4 (노드 내) PP = 4 (노드 간) DP = 4 ZeRO Stage = 1
  * Micro-batch size를 조정해 파이프라인 버블<sup>pipeline bubble</sup>을 최소화하는 것이 좋습니다.
* **100B-200B 모델 추천 구성 (512 GPU / 64 노드):**
  * TP = 8 (노드 내) PP = 8 (노드 간) DP = 8 ZeRO Stage = 1 Micro-batch size = 4-8 Global batch size = 1024-2048

#### MoE Models

<table><thead><tr><th width="104.015625">모델 크기</th><th width="120.06640625">권장 GPU</th><th>베이스라인</th><th>확장(용량/스루풋↑)</th></tr></thead><tbody><tr><td><strong>30B</strong></td><td>16–32</td><td>DP2×TP1×PP2×EP4</td><td>DP2×TP2×PP2×EP4</td></tr><tr><td><strong>70B</strong></td><td>32–64</td><td>DP2×TP2×PP2×EP4</td><td>DP2×TP2×PP2×EP8 </td></tr><tr><td><strong>200B</strong></td><td>128–256</td><td>DP2×TP4×PP4×EP4</td><td>DP2×TP4×PP4×EP8 </td></tr><tr><td><strong>500B</strong></td><td>512–1024</td><td>DP2×TP8×PP8×EP4</td><td>DP2×TP8×PP8×EP8</td></tr><tr><td><strong>700B</strong></td><td>1024–1536</td><td>DP2×TP8×PP8×EP8</td><td>DP3×TP8×PP8×EP8</td></tr></tbody></table>

* **<= 30B 모델 추천 구성 (16 GPU / 2 노드\~ 32 GPU / 4 노드)**
  * 16 GPU: DP2 × TP1 × PP2 × EP4 = 16
  * 32 GPU: DP2 × TP2 × PP2 × EP4 = 32
  * 비고: EP를 동일 노드에 최대한 적용(E/GPU 2–4)해서 노드 간 all-to-all 최소화.
* **<= 70B 모델 추천 구성 (32 GPU / 4 노드\~ 64 GPU / 8 노드)**
  * 32 GPU: DP2 × TP2 × PP2 × EP4 = 32
  * 64 GPU: DP2 × TP2 × PP2 × EP8 = 64
  * 비고: FSDP/ZeRO-3 기본, PP 마이크로배치 ≥ 스테이지 수 권장
* **<= 200B 모델 추천 구성 (128 GPU / 16 노드\~ 256 GPU / 32 노드)**
  * 128 GPU: DP2 × TP4 × PP4 × EP4 = 128
  * 256 GPU: DP2 × TP4 × PP4 × EP8 = 256
  * 비고: 계층형 EP all-to-all(노드내→노드간) 강제, overlap 필수
  * 전문가 수가 많으면 EP size를 더 크게 설정
  * Qwen 3 MoE 예시 (235B 파라미터, 22B 활성 파라미터)
    * DP = 2, TP = 4 (dense 레이어), PP = 2 , EP = 16
* **<= 500B 모델 추천 구성 (512 GPU / 64 노드\~ 1,024 GPU / 128 노드)**
  * 512 GPU: DP2 × TP8 × PP8 × EP4 = 512
  * 1,024 GPU: DP2 × TP8 × PP8 × EP8 = 1024
  * 비고: TP/EP 모두 NVSwitch 도메인에 고정. EFA/IB는 PP/DP 경로 위주
* **700B 모델 추천 구성 (1,024 GPU / 128 노드 \~ 1,536 GPU / 192노드)**
  * 1,024 GPU: DP2 × TP8 × PP8 × EP8 = 1024
  * 1,536 GPU: DP3 × TP8 × PP8 × EP8 = 1536
  * 비고: 전문가 불균형 모니터링 강화
  * DeepSeek-V3 MoE 예시 (671B 파라미터, 37B 활성 파라미터)
    * DP = 4, TP = 8 (dense 레이어), PP = 4-8, EP = 32-64
    * MLA(Multi-head latent attention)로 KV cache 크기를 기존 대비 1/8로 축소
    * Fine-grained expert parallelism 활용



## 3. 트러블슈팅&#x20;

***

### **3.1. GPU CUDA Out of Memory**

**배치 크기 줄이기 및 Gradient checkpointing 활성화**

```bash
# Micro-batch size 절반으로
--micro-batch-size 2  # was 4

# Gradient checkpointing 활성화
--checkpoint-activations
```

**ZeRO stage 2로 변경**

```python
"zero_optimization": {
    "stage": 2  # was 1
}
```

**CPU offload 활성화 및 TP/PP 크기 증가**

```python
# CPU offload 활성화
"zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
        "device": "cpu"
    }
}

# TP/PP size 증가
--tensor-model-parallel-size 16  # was 8
```

### 3.2. 통신 병목

**GPU 활용률이 낮을 때 (30% 이하)**

```python
# 프로파일링
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, 
                profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    train_step()

# 통신 시간 확인
print(prof.key_averages().table(
    sort_by="cuda_time_total", 
    row_limit=10
))
```

**All-reduce 최적화**

```json
{
   "zero_optimization": {
     "overlap_comm": true,
     "contiguous_gradients": true,
     "reduce_bucket_size": 500000000
   }
}
```

**Pipeline bubble 줄이기**

```bash
# Micro-batch 수 증가
NUM_MICROBATCHES=$((PP_SIZE * 4))
 
# Virtual pipeline 활성화
--virtual-pipeline-model-parallel-size 2
```

### 3.3. MoE Load Imbalance

**모니터링**

```python
# Expert utilization 추적
def log_expert_stats(router_logits):
    expert_counts = router_logits.argmax(dim=-1).bincount()
    print(f"Expert usage: {expert_counts}")
    print(f"Std dev: {expert_counts.std().item()}")
    print(f"Max/Min ratio: {expert_counts.max()/expert_counts.min()}")
```

**Load balancing loss 및 capacity factor 조정:**

```python
# Loss coefficient 증가
moe_aux_loss_coeff = 0.01  # 기본값
moe_aux_loss_coeff = 0.05  # 불균형 심할 시

# Token drop 줄이기
capacity_factor = 1.25  # 기본값
capacity_factor = 1.5   # Drop rate 높을 시
```

**Expert initialization 개선**

```python
# 전문가 가중치 다양하게 초기화
for expert in experts:
    expert.apply(lambda m: init_with_variance(m, variance=0.02))
```

### 3.4. 수렴 문제 (**Loss가 spike하거나 NaN 발생)**

**Gradient norm 체크**

```python
# Gradient clipping 강화
--clip-grad 0.5  # was 1.0

# Gradient scaling 조정
"fp16": {
  "initial_scale_power": 12  # was 16
}
```

**Learning rate 재조정**

```python
# LR 줄이기
--lr 3.0e-5  # was 6.0e-5

# Warmup 늘리기
--lr-warmup-iters 2000  # was 500
```

**Numerical stability**

```python
# BF16 사용 (FP16보다 안정적)
--bf16

# LayerNorm epsilon 증가
layernorm_epsilon = 1e-5  # was 1e-6
```



## References

* [DeepSpeed Documentation](https://www.deepspeed.ai/)
* [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
* [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/)
* [Hugging Face Parallelism Guide](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many)
* AWS ML Training Reference Architectures: [https://github.com/aws-samples/awsome-distributed-training](https://github.com/aws-samples/awsome-distributed-training)
* EFA 개요/OS-bypass & libfabric: [https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
* aws-ofi-nccl 깃허브: [https://github.com/aws/aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl)
* EFA+NCCL 설정 AWS 문서 (드라이버, GDRCopy, EFA, NCCL 테스트: [https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html)
