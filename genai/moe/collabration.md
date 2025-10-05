# ML 엔지니어와 인프라 엔지니어 간 분산 훈련 협업 가이드 및 체크리스트

대규모 MoE 모델은 GPU의 연산 자원뿐 아니라 **GPU 간 통신 효율**, **메모리·토폴로지 설계**, **체크포인트 I/O** 등 복합적인 인프라 요인에 의해 성능이 좌우됩니다. 특히 AWS HyperPod, ParallelCluster 같은 클라우드 기반 HPC 환경에서는, 모델 엔지니어(데이터 과학자 및 ML 엔지니어와 인프라/플랫폼 엔지니어가 **초기 설정부터 장애 대응까지 긴밀히 협업**해야 합니다. 본 가이드는 그 협업 과정에서 공통 언어를 만들고, **역할 간 경계에서 자주 발생하는 성능 저하·커뮤니케이션 오류를 예방**하기 위한 지침을 제공합니다.



## 1. 협업 가이드

***

### **1.1. 기본 협업 원칙**

<table><thead><tr><th width="162.22265625">구분</th><th>ML 측 (ML 엔지니어 / 데이터 과학자)</th><th>인프라 측 (인프라 엔지니어 / 플랫폼 엔지니어)</th></tr></thead><tbody><tr><td><strong>관점</strong></td><td>모델의 학습 효율, 수렴, 로스 곡선, 분산 효율</td><td>클러스터 자원 최적화, 통신 지연 최소화, 안정성</td></tr><tr><td><strong>주요 관심사</strong></td><td>모델 병렬 전략 (EP, TP, PP), Batch Size, Gradient Sync</td><td>네트워크 패브릭(EFA/IB), NCCL 설정, 스토리지 I/O</td></tr><tr><td><strong>공유 정보</strong></td><td>모델 구성, expected FLOPs, 메모리 요구량</td><td>실제 TFLOPS, NCCL 통신 로그, GPU Utilization</td></tr><tr><td><strong>커뮤니케이션 목표</strong></td><td>“왜 느린가?”를 재현 가능하게 설명</td><td>“어디서 병목이 생겼는가?”를 가시화</td></tr></tbody></table>

### **1.2. 협업을 위한 공통 기술 언어**

ML 측은 EP·DP·SDP·TP·PP 등 병렬 설정을 명확히 문서화하고, 인프라 측은 이를 실제 GPU mapping(토폴로지) 매칭시켜야 합니다.

#### **핵심 파라미터**

| 항목                            | ML 측 의미              | 인프라 측 의미                   |
| ----------------------------- | -------------------- | -------------------------- |
| **EP (Expert Parallelism)**   | MoE 전문가(Expert) 분할 수 | GPU 간 All-to-All 통신 토폴로지   |
| **DP(Data Parallelism)**      | 데이터 배치               | GPU 간 All-Reduce 통신 토폴로지   |
| **TP (Tensor Parallelism)**   | 텐서 분할 비율             | intra-node NVLink 연결 구조    |
| **PP (Pipeline Parallelism)** | Layer 분할 단계          | stage-to-stage GPU mapping |
| **GSP (Gradient Shard/Sync)** | 그래디언트 통신 방식          | NCCL Ring/Tree/AllToAll 경로 |
| **Activation Checkpointing**  | GPU 메모리 절약 기법        | CPU 메모리 및 IO 부하 고려         |
| **Offload Ratio**             | CPU→GPU 데이터 이동 비율    | PCIe/InfiniBand 대역폭 모니터링   |

### **1.3. 훈련 전 협의 단계 (Pre-training Alignment)**

#### **모델 구조 및 통신 패턴 정의**

* ML 엔지니어/데이터 과학자는 다음 항목을 **명시적으로 전달**해야 합니다.
  * Expert 수 및 Expert 당 GPU 수
  * MoE 라우팅 빈도 (e.g., top-1, top-2 gating)
  * All-to-All 빈도 및 TensorFusion 여부
  * 평균 Batch 크기 / Sequence 길이 / FP16 vs BF16
* 인프라 엔지니어는 이 정보를 기반으로 다음을 **설계 및 피드백**해야 합니다.
  * EP/DP/TP/PP 매핑 계획
  * EFA 네트워크 채널 구성 (EFA Multi-NIC, RDMA path)
  * NCCL/OFI tuning 변수 (NCCL\_BUFFSIZE, NCCL\_P2P\_NET\_CHUNKSIZE, FI\_PROVIDER=efa)

#### **환경 변수 표준화**

MML 측 측과 인프라 측간 불일치를 최소화하려면,  아래 예시와 같은 공통 환경 변수 템플릿을 공유·관리하는 것이 좋습니다. AWS HyperPod나 ParallelCluster에서는 이러한 기본 설정이 되어 있으므로 job-level override 없이 관리할 수 있습니다.

```sh
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_PROTO=LL128
export NCCL_BUFFSIZE=67108864
export NCCL_P2P_NET_CHUNKSIZE=4194304
export NCCL_NET_GDR_LEVEL=PHB
```

### **1.4. 훈련 중 커뮤니케이션 (During Training)**

#### **주요 모니터링 지표 공유**

| 지표                           | ML 측                      | 인프라 측                             |
| ---------------------------- | ------------------------- | --------------------------------- |
| **Throughput (samples/sec)** | 모델 성능 판단                  | 네트워크 병목 판단 기준                     |
| **GPU Utilization (%)**      | 학습 효율                     | scheduling / CUDA stream 병목 추정    |
| **NCCL time breakdown**      | collective op 비중 분석       | EFA/Ethernet path 이상 탐지           |
| **CUDA OOM 비율**              | batch size / optimizer 문제 | memory fragmentation 원인 분석        |
| **Loss Curve stability**     | 모델 tuning                 | node 간 clock drift / 통신 delay 가능성 |

#### **공통 로그 지표**

| 로그 파일               | 설명                      | 활용 주체    |
| ------------------- | ----------------------- | -------- |
| NCCL\_DEBUG=INFO 로그 | collective call latency | 인프라      |
| torch.profiler 결과   | operator-level latency  | ML       |
| fi\_info -p efa     | RDMA 상태 확인              | 인프라      |
| dmesg / efa-dmesg   | driver-level 오류 탐지      | 인프라      |
| nvidia-smi dmon     | GPU 온도, power draw      | 인프라 + ML |

커뮤니케이션 예시:

* ML 엔지니어 “Step 300에서 AllReduce latency가 2배 상승했습니다. NCCL log에서 ring 0의 sync 지연이 보입니다.”
* 인프라 엔지니어: “같은 시점에 fi\_info 출력에서 EFA 연결 재시도(retry)가 발생했습니다. SRD congestion 가능성이 있습니다.”

### **1.5. 장애 대응 및 튜닝 시 역할 분담**

<table><thead><tr><th width="191.62109375">상황</th><th>ML 측</th><th>인프라 측</th></tr></thead><tbody><tr><td><strong>OOM 발생</strong></td><td>Batch size / activation checkpoint 조정</td><td>HugePage, NUMA 설정 점검</td></tr><tr><td><strong>Throughput 급감</strong></td><td>dataloader / sharding 점검</td><td>EFA 드라이버, MTU, NCCL 버퍼 확인</td></tr><tr><td><strong>NCCL hang</strong></td><td>torch.distributed init 재시도</td><td>EFA provider log, network retry trace</td></tr><tr><td><strong>노드 간 불균형</strong></td><td>expert load imbalance</td><td>RDMA 경로, IRQ balance 점검</td></tr><tr><td><strong>loss spike / 불안정</strong></td><td>FP precision 확인</td><td>clock drift / time sync 검증</td></tr></tbody></table>

장애 원인이 모호할 경우, dmesg, efa-dmesg, nccl.log, torch.profiler 를 시간순으로 정렬하여 cross-check 하는 절차를 두는 것이 좋습니다.

### **1.6. 훈련 후 리뷰 및 지속 개선**

* **훈련 완료 후**, 두 팀이 다음 데이터를 함께 리뷰합니다.
  * 평균 GPU utilization 및 각 collective op latency
  * MoE routing imbalance (active expert ratio)
  * SRD retransmission / EFA congestion 통계
  * 노드별 throughput 및 checkpoint save 시간
*   이상 패턴이 반복되면, HyperPod cluster template이나 ParallelCluster config에 feedback loop를 추가해

    차기 실험 시 자동 반영되도록 합니다.

커뮤니케이션 예시:

* “이번 세션에서 Expert 간 imbalance가 심해 NCCL AllToAll latency가 평균 30% 증가했습니다. 다음 세션에서는 EP=8에서 EP=16으로 분할해 MoE 라우팅 경로를 더 고르게 분산해 볼까요?”

### **1.7. 결론 및 권장 운영 프로세스**

1. **훈련 전:** 모델 병렬 전략(EP/DP/TP/PP) 및 통신 패턴을 인프라 팀에 상세히 문서화합니다.
2. **훈련 중:** 공통 로그(Throughput, NCCL, fi\_info)를 기반으로 주기적 sync 미팅을 진행합니다.
3. **훈련 후:** bottleneck 구간 분석 결과를 cluster config(YAML, Slurm template)에 피드백합니다.
4.  **자동화:** SageMaker HyperPod의 Slurm job template 또는 ParallelCluster의 post\_install 스크립트에

    환경 변수 및 로깅 설정을 자동 삽입합니다.



## 2. **MoE 훈련 커뮤니케이션 체크리스트**

***

### **✅ 2.1. 훈련 전 (Pre-Training Alignment)**

<table><thead><tr><th>항목</th><th width="76.1484375">담당</th><th>구체적 내용</th><th>전달 방식 / 도구</th></tr></thead><tbody><tr><td><strong>모델 병렬 전략</strong></td><td>ML</td><td>EP, DP, TP, PP 비율 / Expert 수 / Routing(top-1, top-2) 명시</td><td>문서화(Confluence, Notion 등)</td></tr><tr><td><strong>데이터셋 / 시퀀스 길이</strong></td><td>ML</td><td>Batch size, seq_len, token 수, 샘플 개수</td><td>Slack / YAML 공유</td></tr><tr><td><strong>통신 패턴</strong></td><td>ML</td><td>AllReduce / AllToAll 빈도, MoE gating 빈도</td><td>설명서 첨부</td></tr><tr><td><strong>GPU 메모리 요구량</strong></td><td>ML</td><td>최소 / 평균 / 최대 usage 예측</td><td>초기 프로파일 결과</td></tr><tr><td><strong>EFA 설정 계획</strong></td><td>인프라</td><td>EFA NIC 수, Multi-Rail 설정, MTU(9001)</td><td>cluster config YAML</td></tr><tr><td><strong>Libfabric / NCCL 버전</strong></td><td>인프라</td><td>libfabric >=1.17, aws-ofi-nccl 버전</td><td>설치 로그</td></tr><tr><td><strong>환경 변수 표준화</strong></td><td>공동</td><td>FI_PROVIDER=efa, NCCL_BUFFSIZE, NCCL_PROTO 등</td><td>.env 파일</td></tr><tr><td><strong>보안 그룹 / 서브넷 확인</strong></td><td>인프라</td><td>모든 노드 동일 서브넷, SG 내 상호 통신 허용</td><td>AWS CLI 점검</td></tr></tbody></table>

### **🚀 2.2. 훈련 중 (During Training Execution)**

<table><thead><tr><th>항목</th><th width="75.83203125">담당</th><th>주요 지표 / 로그</th><th>검증 기준 / 액션</th></tr></thead><tbody><tr><td><strong>Throughput(samples/s)</strong></td><td>ML</td><td>학습 로그 / wandb</td><td>±5% 이상 변화 시 통신 병목 의심</td></tr><tr><td><strong>GPU Utilization</strong></td><td>인프라</td><td>nvidia-smi dmon, DCGM exporter</td><td>90% 미만 지속 시 NCCL 점검</td></tr><tr><td><strong>NCCL 통신 로그</strong></td><td>인프라</td><td>NCCL_DEBUG=INFO</td><td>AllToAll latency, retransmit 여부</td></tr><tr><td><strong>fi_info -p efa 결과</strong></td><td>인프라</td><td>provider=efa, EP_RDM 확인</td><td>fallback 없는지 확인</td></tr><tr><td><strong>OOM 발생 여부</strong></td><td>ML</td><td>training log</td><td>checkpoint / gradient sharding 조정</td></tr><tr><td><strong>SRD congestion 이벤트</strong></td><td>인프라</td><td>efa-dmesg, CloudWatch metrics</td><td>Nitro NIC 재시작 여부</td></tr><tr><td><strong>Collective op 분포</strong></td><td>ML</td><td>torch.profiler / DeepSpeed log</td><td>bottleneck op 식별</td></tr><tr><td><strong>모델 수렴 안정성</strong></td><td>ML</td><td>loss curve, gradient variance</td><td>통신 delay에 따른 불안정 탐지</td></tr><tr><td><strong>노드간 clock drift</strong></td><td>인프라</td><td>NTP sync 확인</td><td>10 ms 이상 차이 시 재동기화</td></tr></tbody></table>

### **🧩 2.3. 훈련 후 (Post-Training Review)**

<table><thead><tr><th>항목</th><th width="75.703125">담당</th><th>점검 포인트</th><th>활용 방안</th></tr></thead><tbody><tr><td><strong>GPU Utilization 평균/분산</strong></td><td>공동</td><td>노드별 비교</td><td>imbalance 분석</td></tr><tr><td><strong>AllReduce / AllToAll 평균 지연</strong></td><td>인프라</td><td>NCCL 로그 요약</td><td>SRD 튜닝 파라미터 개선</td></tr><tr><td><strong>Expert Load Distribution</strong></td><td>ML</td><td>MoE routing 결과</td><td>gating imbalance 보정</td></tr><tr><td><strong>Checkpoint 저장 속도</strong></td><td>인프라</td><td>I/O latency</td><td>FSx / EBS IOPS 조정</td></tr><tr><td><strong>SRD Retransmission 통계</strong></td><td>인프라</td><td>Nitro NIC metric</td><td>ECMP / MultiPath 최적화</td></tr><tr><td><strong>Loss 안정성</strong></td><td>ML</td><td>수렴 곡선</td><td>optimizer / precision 튜닝</td></tr><tr><td><strong>성능 개선 제안</strong></td><td>공동</td><td>tuning 요약서</td><td>cluster config 반영</td></tr><tr><td><strong>버전 관리</strong></td><td>인프라</td><td>efa-version, libfabric-version 기록</td><td>차기 실험 reproducibility 확보</td></tr></tbody></table>

### **📊 2.4. 참고: 추천 모니터링 도구 세트**

<table><thead><tr><th width="141.515625">범주</th><th>도구</th><th>설명</th></tr></thead><tbody><tr><td><strong>GPU 상태</strong></td><td>nvidia-smi dmon, DCGM exporter</td><td>온도, power, utilization</td></tr><tr><td><strong>NCCL 통신</strong></td><td>NCCL_DEBUG=INFO, NCCL_TOPO_DUMP_FILE</td><td>topology, latency trace</td></tr><tr><td><strong>EFA 상태</strong></td><td>fi_info, efa-dmesg, CloudWatch EFA metrics</td><td>provider 확인, SRD event</td></tr><tr><td><strong>학습 로깅</strong></td><td>wandb, tensorboard, torch.profiler</td><td>throughput / loss 시각화</td></tr><tr><td><strong>네트워크 통계</strong></td><td>ethtool -S, iperf3 --efa</td><td>링크 대역폭, packet drop 확인</td></tr></tbody></table>

### **⚙️ 2.5. 권장 협업 프로세스 요약**

1. **공통 문서화:** 모든 MoE 설정(EP/TP/PP, Batch, Routing)을 YAML/Markdown으로 관리
2. **일일 모니터링 회의:** 훈련 중 주요 지표(Throughput, Latency) 공유 (예: 스탠드업 5분 미팅)
3. **주간 튜닝 리포트:** Infra팀이 NCCL/EFA 로그를 요약 → ML팀에 피드백
4. **Config 자동화:** HyperPod job template / ParallelCluster post\_install 스크립트에 환경 변수 통합
5. **회고 루프:** 훈련 종료 후 Bottleneck 분석 → 다음 실험 config로 반영
