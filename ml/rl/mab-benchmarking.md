# MAB Algorithm Benchmarking

## 0. 개요

### Objective

* MAB 기본 알고리즘들의 성능\(convergence 속도, latency\) 측정

### **방법**

* convergence: experiments의 평균 reward값과 최적의 arm을 선택할 확률을 plotting
* latency: agent 초기화 부분을 제외한 실제 MAB 알고리즘 구동부의 속도만 측정

```text
%matplotlib inline
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import bandits as bd
from bandits import kullback
```

## **1. 하이퍼파라메터 설정**

* **Number of Arms**: arm의 갯수, 즉 k
* **Number of Trials**: Number of Iterations \(Time steps\)
* **Number of Experiments**: Converge 척도 실험; 각자 다른 세팅의\(default:random\) 500개의 k arms들을 구동시켜 평균을 취하기 위한 목적임

```text
n_arms = 10
n_trials = 1000
n_experiments = 200
```

## **2. 𝜖-greedy\(epsilon-greedy\)**

* 개요: 𝜖 확룰로 random하게 탐색하고 1 − 𝜖의 확률로 greedy하게 가장 좋은 arm을 선택하는 알고리즘으로 𝜖이 크면 탐색의 비중이 높아지고 𝜖이 작으면 획득의 비중이 높아짐
* 비교 알고리즘
  * Random: 𝜖=1 인 경우
  * Greedy: 𝜖=0 인 경우
  * EpsilonGreedy \(w/ 𝜖=0.01\): 𝜖=0.01 인 경우
  * EpsilonGreedy \(w/ 𝜖=0.1\): 𝜖=0.01 인 경우 \(일반적으로 쓰이는 세팅값이나 환경에 따라 조절 필요\)
* 실험 결과 \(Converge Ratio\):
  * Random: Average Reward와 Optimal Action의 선택 확률이 매우 낮음
  * Greedy: Average Reward가 Random 대비 높으나 Optimal Action의 선택 확률이 일정 시점 이후에 증가하지 않음
  * EpsilonGreedy \(w/ 𝜖=0.01\): Optimal Action의 선택 확률이 점진적으로 증가하지만 수렴 속도가 더딤
  * EpsilonGreedy \(w/ 𝜖=0.1\): Optimal Action의 선택 확률이 나머지 셋 대비 가장 좋으며 증가 속도도 빠름
* 실험 결과 \(Elapsed Time\):
  * 4가지 Policy 모두 큰 차이는 없으나 Random의 경우 속도가 약간 떨어짐

```text
bandit = bd.GaussianBandit(n_arms)
agents = [
    bd.Agent(bandit, bd.RandomPolicy()),
    bd.Agent(bandit, bd.GreedyPolicy()),
    bd.Agent(bandit, bd.EpsilonGreedyPolicy(0.01)),
    bd.Agent(bandit, bd.EpsilonGreedyPolicy(0.1)),
]
env = bd.Environment(bandit, agents, label='Epsilon Greedy')
scores, optimal = env.run(n_trials, n_experiments)
env.plot_results(scores, optimal)
```

![](../../.gitbook/assets/untitled%20%287%29.png)

## **3. UCB\(Upper Confidence Bound\)**

* 개요: Upper Confidence Bound\(이하 UCB\)를 설정하여 reward에 대한 불확실성을 고려
  * As-is: 주사위를 여러 번 던졌을 때 기대값은 3.5이지만, 두 번만 던졌을 때 눈금이 1, 3이 나오면 기대값이 2가 나오므로 실제 기대값과 편차가 심함
  * To-be: 주사위를 두 번만 던졌을 때 \[2, 5.2\]의 범위로 Confidence Interval을 정하고 횟수가 증가할 수록 Confidence Interval을 감소시켜 불확실성을 줄임
* 비교 알고리즘
  * EpsilonGreedy \(w/ 𝜖=0.1\)
  * EpsilonGreedy \(w/ 𝜖=0.1, Initial Value=5\): Initial Value를 높게 설정 시 초기 Iteration에서 탐색 비중이 올라가기 때문에 최적 Action의 선택 빈도가 늘어남
  * UCB \(w/ c\): c는 Confidence level를 조절하는 하이퍼파라메터로 값이 높을수록 탐색 빈도 증가
    * 직관적 이해1: 계속 특정 arm만 선택하면 UCB term이 작아지므로 Confidence Interval이 감소하여 탐색 빈도 감소 \(𝑁𝑡\(𝑎\) 가 증가하므로\)
    * 직관적 이해2: log의 역할은 UCB decay로 점차 Confidence Interval을 감소시키는 역할
    * 직관적 이해3: 잘 선택되지 않은 arm은 Confidence Interval이 감소하여 탐색 빈도 증가

### **3.1. Gaussian Bandits**

* 실험 결과 \(Converge Ratio\):
  * EpsilonGreedy \(w/ 𝜖=0.1, Initial Value=5\): Iteration 극초기에 % Optimal Action 증가 속도가 셋 중 가장 빠름
  * UCB: Average Reward 및 Optimal Action의 선택 확률이 셋 중 가장 높지만 중간중간 성능이 저하되는 구간 발생
* 실험 결과 \(Elapsed Time\):
  * UCB의 경우 𝜖-greedy 대비 약 1.5배 정도 지연됨

```text
bandit = bd.GaussianBandit(n_arms)
agents = [
    bd.Agent(bandit, bd.EpsilonGreedyPolicy(0.1)),
    bd.Agent(bandit, bd.EpsilonGreedyPolicy(0.1), prior=5),
    bd.Agent(bandit, bd.UCBPolicy(1))
    #bd.Agent(bandit, bd.MOSSPolicy())    
]
env = bd.Environment(bandit, agents, label='Epsilon Greedy and UCB')
scores, optimal = env.run(n_trials, n_experiments)
env.plot_results(scores, optimal)
```



![](../../.gitbook/assets/untitled-1%20%287%29.png)

### 3.2. Bernoulli Bandits

```text
bandit = bd.BernoulliBanditNumpy(n_arms)
agents = [
    bd.Agent(bandit, bd.EpsilonGreedyPolicy(0.1)),
    bd.Agent(bandit, bd.EpsilonGreedyPolicy(0.1), prior=5),
    bd.Agent(bandit, bd.UCBPolicy(1))
]
env = bd.Environment(bandit, agents, label='Epsilon Greedy and UCB')
scores, optimal = env.run(n_trials, n_experiments)
env.plot_results(scores, optimal)
```

![](../../.gitbook/assets/untitled-2%20%284%29.png)

## **4. UCB Variants**

* 개요: 다양한 UCB 알고리즘들에 대한 성능 평가

### 4.1. Bernoulli Bandits

```text
bandit = bd.BernoulliBanditNumpy(n_arms)
agents = [
    bd.Agent(bandit, bd.UCBPolicy()),
    bd.Agent(bandit, bd.UCBVPolicy(2, 0.1)),
    bd.Agent(bandit, bd.UCBTunedPolicy()),
    bd.Agent(bandit, bd.KLUCBPolicy(klucb=kullback.klucbBern)),
    bd.Agent(bandit, bd.MOSSPolicy()),
    bd.BetaAgentUCB(bandit, bd.GreedyPolicy())
]
env = bd.Environment(bandit, agents, label='UCB Variants (Bernoulli Bandits)')
scores, optimal = env.run(n_trials, n_experiments)
env.plot_results(scores, optimal)
```

### 4.2. Binomial Bandits

```text
bandit = bd.BinomialBandit(n_arms, n=5, t=8*n_trials)
agents = [
    bd.Agent(bandit, bd.UCBPolicy()),
    bd.Agent(bandit, bd.UCBVPolicy(2, 0.1)),
    bd.Agent(bandit, bd.UCBTunedPolicy()),
    bd.Agent(bandit, bd.KLUCBPolicy(klucb=kullback.klucbBern)),
    bd.Agent(bandit, bd.MOSSPolicy()),
    bd.BetaAgentUCB(bandit, bd.GreedyPolicy())
]
env = bd.Environment(bandit, agents, label='UCB Variants (Binomial Bandits)')
scores, optimal = env.run(n_trials, n_experiments)
env.plot_results(scores, optimal)
```

## **5. TS\(Thompson Sampling\)**

* 개요: Beta 분포를 prior로 가정하고 Bandit의 분포를 베르누이 분포나 이항 분포의 형태를 가지는 likelihood로 가정하여 확룰 분포를 모델링하는 방법
* 비교 알고리즘
  * EpsilonGreedy \(w/ 𝜖=0.1, Initial Value=5\)
  * UCB
  * TS

### **5.1. pymc3 라이브러리**

* 원본 소스코드로 theano사용으로 인해 non-GPU 환경에서 속도가 느림
* 실험 결과 \(Converge Ratio\):
  * EpsilonGreedy \(w/ 𝜖=0.1, Initial Value=5\): 안정적으로 수렴하지만 일정 Iteration 이상에서 수렴속도가 떨어짐
  * UCB: Average Reward와 Optimal Action의 선택 확률이 Iteration이 반복될수록 꾸준히 증가하지만 중간중간 성능이 저하되는 구간 발생
  * TS: 1000번 이내의 time step에서는 가장 우수한 성능을 보이나 5000회, 10000회 등 long run으로 갈수록 UCB에 비해 불리할 수 있음
* 실험 결과 \(Elapsed Time\):
  * TS의 알고리즘 로직은 간단하나 pymc3 라이브러리의 Beta 확률 분포 모델링에서 로드가 많이 걸림

### **5.2. numpy 라이브러리**

* pymc3 라이브러리를 활용하지 않고 numpy로만 구현한 코드
* Non-caching과 caching으로 나누어서 running time 비교 \(pymc3 라이브러리 기반 구현은 caching 사용\)
  * Non-caching시, reward값의 prior 분포는 매 time step마다 \[num\_arms x 1\] 벡터로 계산
  * Caching시, reward값의 prior 분포는 \[num\_trials x num\_arms\] 행렬로 미리 계산하여 매 time step마다 \[num\_arms x 1\] 벡터 리턴
* 실험 결과, Caching 사용 시의 running time이 non-caching 대비 약 3배 이상 더 빠름

```text
bandit = bd.BernoulliBanditNumpy(n_arms)
agents = [
    bd.Agent(bandit, bd.EpsilonGreedyPolicy(0.1), prior=5),
    bd.Agent(bandit, bd.UCBPolicy(1)),
    bd.BetaAgentNumpy(bandit, bd.GreedyPolicy())
]

env = bd.Environment(bandit, agents, label='Epsilon Greedy vs. UCB vs. Thompson Sampling')
scores, optimal = env.run(n_trials, n_experiments)
env.plot_results(scores, optimal)
```

![](../../.gitbook/assets/untitled-3%20%281%29.png)

