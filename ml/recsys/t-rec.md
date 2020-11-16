---
description: PoC 진행 전 기술 검토를 위한 논문 리뷰
---

# T-REC\(Towards Accurate Bug Triage for Technical Groups\) 논문 리뷰

## 1. Preliminaries

### BM25 \(Okapi BM25\)

* TF-IDF 기반 ranking 함수로 검색어와 문서의 관련도를 평가하는 방법
* ElasticSearch에서 사용하고 있음
* BM = Best Matching
* 수식이 복잡해 보이지만, TF-IDF의 수식에 Document Length Normalization과 Saturation function을 추가한 것일 뿐임

$$
\text{score}(D,Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \dfrac{ f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1-b+b \cdot \dfrac{|D|}{\text{avgdl}})}
$$

$$
\text{IDF}(q_i) = \log \dfrac{N - n(q_i) + 0.5}{n(q_i) + 0.5}
$$

* $$q_i$$:  $$i$$번째 쿼리 토큰
* $$f(q_i, D)$$: TF\(Term Frequency\); 단어의 출현 빈도로 쿼리의 $$i$$번째 쿼리 큰이 문서 D에 얼마나 자주 나타나는가의 의미
* $$\text{IDF}(q_i)$$: i번째 쿼리 토큰에 대한 the inverse document frequency
  * N: 문서의 총 개수
  * 자주 등장하는 단어에 대해 페널티를 부여하여 지나치게 높은 가중치를 가지게 되는 것을 방지
  * 예: 불용어 \(stopword\)
* $$\dfrac{|D|}{\text{avgdl}}$$ :  문서의 총 단어 / 모든 문장의 평균 단어 수
  * 해당 문서가 평균적인 문서 길이에 비해 얼마나 긴지 고려하며, 평균 대비 긴 문서는 penalize됨
  * 예: 긴 문서\(예: 백서\)에서 특정 단어가 언급되는 것보다 짧은 문서\(예: 트윗\)에서 특정 단어가 언급되는 것이 더 의미가 있음
* $$b$$: 길이에 대한 가중치로 0에 가까울수록 문서 길이의 가중치가 감소 \(보통 0.75\)
* $$k_1$$: TF의 saturation을 결정하는 요소 \(보통 1.2~2.0\)
  * 하나의 쿼리 토큰이 문서 점수에 줄 수 있는 영향을 제한함.
  * 직관적으로$$k_1$$은 어떤 토큰이 했을 때, 이전까지에 비해 점수를 얼마나 더 높여주어야 하는가를 결정함.
  * 어떤 토큰의 TF가 $$k_1$$보다 작거나 같으면 TF가 점수에 주는 영향이 빠르게 증가하는데, $$k_1$$번 등장하기 전 까지는 해당 키워드가 문서에 더 등장하게 되면 문서의 BM25 스코어를 빠르게 증가시킨다는 뜻임.
  * 반대로 토큰의 TF가 $$k_1$$보다 크면 TF가 점수에 주는 영향이 천천히 증가하고, 해당 키워드가 더 많이 등장해도 스코어에 주는 영향이 크지 않음.
* BM25F: a modification of BM25 in which the document is considered to be composed from several fields \(such as headlines, main text, anchor text\) with possibly different degrees of importance, term relevance saturation and length normalization.
  * 각 필드에 대한 Local Importance를 aggregation

## 2. Pre-processing

### Category Normalization

* 데이터 입력 폼이 중구난방이기 때문에 Levenshitein distnace가 서로 유사한 단어들을 grouping
* 문장 부호\(Punctunation\), 불용어\(Stopwords\), 숫자\(Digits\) 등의 불필요한 정보를 필터링으로 제거
* 예: _Others-Booting → boot\_other, Others-Boot → boot\_other, Voice Call → call\_voic, Voice Call or Call → call\_voic_

### Language-Related Processing

* 사용자는 자연어와 비자연어를 혼합하는 경향이 있음
* 비정형 데이터 입력에서 structural software engineering 데이터 추출을 위해 오픈 소스 툴인 infoZilla 사용
* 언어는 일부 TG를 구별하는 데 효율적이므로 오픈 소스 language detector인 CLD2Owners 사용
  * Compact Language Detector 2 \([https://github.com/CLD2Owners/cld2](https://github.com/CLD2Owners/cld2)\)

## 3. T-REC

An auxiliary SW Project Management system

![](../../.gitbook/assets/untitled%20%282%29.png)

### Model

* VSM\(Vector space model\)과 IR 알고리즘 결합
* VSM은 FastText 사용
* Returns a rank of tuples of similar issues $$R = {(D_1, S_1), (D_2, S_2), ..., (D_K, S_K)}$$ 
  * $$R$$: Relevant Issue, $$S$$: Similarity Score
* Listwise MLR 접근 방식에서 영감을 얻은 aggregation 사용 \(예: ListNet, ListMLE\)
  * Listwise Loss function
  * 궁금증

    * Aggregation function F\_a의 정체는?
    * Pairwise loss function은 왜 고려하지 않았는지?

    ![](../../.gitbook/assets/untitled-1%20%285%29.png) 

### Ranking Function

* BM25F를 개량한 BM25F\_ext 변형; LC25라 하지만 논문에서 자세한 설명을 하고 있지 않음
  * BM25F는 short 쿼리에 적합한 metric으로 short 쿼리는 대체로 duplicate word가 없음
* BM25F\_ext: BM25F에 $$W_Q$$term 추가
  * query term에 대한 출현 빈도를 aggregation

![](../../.gitbook/assets/untitled-2%20%286%29.png)

![](../../.gitbook/assets/untitled-3%20%286%29.png)

![](../../.gitbook/assets/untitled-4%20%284%29.png)

* $$d$$= document, $$q$$=  query, $$t$$= term of a document, $$f$$= field of a document
* $$k_1$$= control the effect of $$TF_Q(q,t)$$ 
* $$k_3$$= contribution of the query term weighting \( $$k_3 = 0$$이면 BM25와 동일\)

## 4. Experiments

### Accuracy

* 2001년 1월부터 2019년 1월까지 Product Lifecycle Management에서 수집한 9.5M 모바일 관련 문제로 구성된 Sidia dataset 사용 \(private dataset\)
  * Sidia: 삼성전자 브라질 연구소
* 하지만 실제 사용 가능한 데이터는 이보다 훨씬 적음
  * Noisy 데이터 다수 존재 \(잘못된 사용자 입력, 리팩토링된 필드, 미사용 레거시 데이터, 자연어/비자연어 혼합한 텍스트 필드\)
  * 많은 closed issue에 버그를 해결한 개발자를 찾을 수 없고 55%는 TG 레이블이 없음
* 실험 시에는 2017년 1월부터 2017년 9월까지 10만개의 문제 샘플링
* Accuracy@k를 평가 지표로 사용
  * 궁금증: 추천 문제에 왜 NDCG, Precision, MRR같은 지표도 같이 고려하면 어떤지?
* T-REC \(LC25F + KNN\), LC25F, KNN, Random Forest, SVM 및 MLP의 6가지 방법 비교
* 아래 표와 같이 T-REC는 Accuracy@1에서 0.509, Accuracy@20에서 0.897 달성

![](../../.gitbook/assets/untitled-5%20%285%29.png)

### Tossing Time

* 기존에 A → B → C → D로 이슈를 토스하고 D가 해결했다면 당연히 D에게 direct toss하는 편이 효율적임
* T-REC으로 테스트 결과, 이에 대한 Tossing time이 대폭 개선
* 가운데 그림은 T-REC에서 추천한 결과를 보여줌 \(Rank@3\)
* 오른쪽 그림은 T-REC의 추천 결과를 적용했을 때의 토스 횟수 저감을 보여줌 \(다만, \(d\)의 그림은 개연성 부족\)

![](../../.gitbook/assets/untitled-6%20%281%29.png)

