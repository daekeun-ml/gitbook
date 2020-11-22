# Introduction

MNIST dataset 클래스 분포는 완벽하게 동일한 비율로 맞춰져 있지만, 실제 데이터는 특정 클래스의 분포가 매우 적은 경우들이 많습니다. 특히 Predictive Analytics에서 가장 많이 활용되는 Tabular 데이터에서 많이 찾아볼 수 있는데, 고객 이탈을 방지하기 위한 churn prediction 이진 분류 문제를 예시로 들어도 실제 이탈한 고객의 비율은 이탈하지 않은 고객 대비 매우 적습니다. \(1:10~1:100\)

이러한 데이터를 그대로 훈련 시에는 다수 클래스에 속한 데이터들의 분포를 위주로 고려하기에 다수 클래스에 속한 데이터에 과적합이 발생하게 되며, 소수 클래스에 속한 데이터는 잘 분류하지 못할 가능성이 높아집니다.

이러한 문제를 해결하기 위해 다수 클래스에 속한 데이터들을 샘플링 기법으로 적게 추출하는 undersampling 기법이나 소수 클래스에 속한 데이터들의 패턴을 파악하여 데이터를 늘리는 oversampling 기법들을 생각해볼 수 있습니다.

그 외에 소수 클래스에 속한 데이터들에 더 큰 가중치를 부여하는 weighting 기법이나, 소수 클래스에 속한 데이터를 잘못 분류 시 penalty를 크게 부여하는 cost-sensitive learning 기법, 다수 클래스에 속한 일부 데이터를 소수 클래스에 속한 데이터 내에서 복원 추출 후 앙상블하는 ensemble sampling 기법도 활용할 수 있습니다.

불균형 클래스 데이터셋에서 일반적으로 사용하는 metric들을 간단히 살펴 보겠습니다. 매우 기본적인 내용이므로, 이미 내용을 알고 있으면 스킵해도 무방합니다.

## Metrics

### ROC Curve

Receiver Operating Characteristic\(수신자 조작 특성\)이라는 이상한 용어 때문에 헷갈릴 것 같아 잠깐 용어의 유래를 언급하겠습니다. 이 용어는 2차 세계 대전 때 "Chain Home" 레이더 시스템의 일부로 영국에서 처음 사용된 개념으로 레이더로 적군 전투기와 신호 잡음\(예: 새\) 판별하기 위해 사용되었습니다.

레이더 범위에 적군 전투기뿐만 아니라 새도 들어오는 경우들이 종종 있는데, 이 때 레이더 정찰병이 경보를 모두 전투기로 판단하면 오보일 확률이 올라가고 경보를 대수롭지 않게 생각해서 무시하면 정작 중요한 때를 놓치게 됩니다. 이에 대한 trade-off를 2차원 좌표\(y축은 TPR; True Positive Ratio, x축은 FPR; False Positive Ratio\)로 나타낸 것이 ROC 곡선입니다. 판별 기준이 정찰병마다 다르기 때문에 각 정찰병의 판별 결과가 달랐지만, 정찰병들의 데이터를 종합하니 곡선 형태가 크게 바뀌지 않다는 것을 알 수 있게 되었고 이는 안정적으로 모델의 성능을 판별하는 지표 중 하나가 되었습니다.

### PR\(Precision-Recall\) Curve

전반적인 모델의 성능을 판별하는 지표로 ROC 곡선이 현재도 널리 쓰이지만, 불균형도가 매우 큰 데이터셋이나 특정 테스트 셋에서의 결과가 중요하다면 PR 곡선도 같이 고려해야 합니다. [The Relationship Between Precision-Recall and ROC Curve 논문](https://www.biostat.wisc.edu/~page/rocpr.pdf)에서 PR 곡선의 필요성에 대한 이유를 아래와 같이 기술하고 있습니다.

> Consequently, a large change in the number of false positives can lead to a small change in the false positive rate used in ROC analysis. Precision, on the other hand, by comparing false positives to true positives rather than true negatives, captures the effect of the large number of negative examples on the algorithm’s performance.

즉, TN이 많다면\(다수 범주에 속한 데이터가 많다면\), FP의 변화량에 비해 FPR의 변화량이 미미합니다.

ROC 곡선은 TN\(True Negative\)에 속한 데이터가 많다면 \(즉, 다수 클래스에 속한 데이터겠죠\), FP\(False Positive\)의 변화량에 비해 FPR의 변화량이 미미합니다

간단한 예시로 1백만 명의 정상인과 100명의 암환자가 포함된 데이터셋에서 암환자를 분류하는 두 개의 모델을 훈련했다고 가정하겠습니다.

* 1번 모델: 100명의 암환자로 검출했는데 실제로 암환자가 90명인 경우
* 2번 모델: 2,000명을 암환자로 검출했는데 실제로 암환자가 90명인 경우

따로 계산하지 않아도 당연히 1번 모델이 더 좋은 모델이겠죠? 그럼 ROC와 PR 기준으로 실제로 계산을 수행해 보겠습니다.

* ROC 기준으로 평가 시,
  * 1번 모델: $$\text{TPR} = 0.9, \; \text{FPR} = (100 - 90) / 1000000 = 0.00001$$
  * 2번 모델: $$\text{TPR} = 0.9, \;\text{FPR} = (2000 - 90) / 1000000 \approx 0.00191$$
  * 두 모델의 FPR 차이는 $$0.00191 - 0.00001 = 0.0019$$입니다.
* PR 기준으로 평가 시,
  * 1번 모델: $$\text{Recall} = 0.9, \; \text{Precision} = 90/100 = 0.9$$
  * 2번 모델: $$\text{Recall} = 0.9, \; \text{Precision} = 90/100 = 0.9$$
  * 두 모델의 Precision 차이는 $0.9 - 0.0045 = 0.855$입니다.
* 불균형 클래스 데이터셋에서 두 모델의 성능 차이를 명확히 파악하려면, PR 커브도 필요하다는 것을 알 수 있습니다.

### AUROC \(Area Under a ROC Curve, aka ROC AUC, AUC\)

ROC 곡선 아래 영역, 즉 TPR과 FPR에 대한 면적을 의미하며, 이 값의 범위는 0~1입니다. 임계값\(threshold\)과 상관 없이 모델의 예측 성능을 정량적으로 알 수 있기에 분류 문제의 metric으로 널리 쓰이고 있습니다.

### AUPRC \(Area Under a PR Curve, aka PR AUC\)

PR 곡선 아래 영역, 즉 Precision과 Recall에 대한 면적을 의미하며, 이 값의 범위는 0~1입니다.

### MCC \(Matthews correlation coefficient\)

F1 점수는 TN을 무시하지만, MCC는 confusion matrix의 4개 값 모두를 고려하므로 4개 값 모두 모두 좋은 예측 결ㅁ과를 얻는 경우에만 높은 점수를 받을 수 있습니다.

$$
{\begin{aligned} \textrm{MCC} = \frac{\text{TP}\cdot\text{TN}-\text{FP}\cdot\text{FN}}{\sqrt{ (\text{TP}+\text{FP})\cdot(\text{TP}+\text{FN})\cdot(\text{TN}+\text{FP})\cdot(\text{TN}+\text{FN}) }}\ \end{aligned}}
$$

MCC는 -1에서 1사이의 값으로 1은 Perfect Prediction, 0은 Random Prediction, -1은 Worst Prediction을 의미합니다. Accuracy, F1 점수, MCC의 결과 비교에 대한 자세한 내용은 아래 링크를 참조하세요.

* [https://github.com/davidechicco/MCC](https://github.com/davidechicco/MCC)



