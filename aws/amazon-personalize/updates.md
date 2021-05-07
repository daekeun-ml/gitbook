# Amazon Personalize Updates\(~2021.04\) 및 FAQ

## 1. Personalize 기능 주요 업데이트 링크

* Batch Recommendation \(2019.11.18\)
  * [https://aws.amazon.com/ko/about-aws/whats-new/2019/11/amazon-personalize-now-supports-batch-recommendations/](https://aws.amazon.com/ko/about-aws/whats-new/2019/11/amazon-personalize-now-supports-batch-recommendations/)
* Context Recommendation \(2019.12.20\)
  * [https://aws.amazon.com/ko/about-aws/whats-new/2019/12/amazon-personalize-supports-contextual-recommendations/](https://aws.amazon.com/ko/about-aws/whats-new/2019/12/amazon-personalize-supports-contextual-recommendations/)
  * [https://aws.amazon.com/blogs/machine-learning/increasing-the-relevance-of-your-amazon-personalize-recommendations-by-leveraging-contextual-information/](https://aws.amazon.com/blogs/machine-learning/increasing-the-relevance-of-your-amazon-personalize-recommendations-by-leveraging-contextual-information/)
* 10x more Item Metadata Fields \(2020.02.07\)
  * [https://aws.amazon.com/ko/about-aws/whats-new/2020/02/amazon-personalize-use-10x-more-item-attributes-improve-relevance-recommendations/](https://aws.amazon.com/ko/about-aws/whats-new/2020/02/amazon-personalize-use-10x-more-item-attributes-improve-relevance-recommendations/)
  * [https://aws.amazon.com/ko/blogs/machine-learning/amazon-personalize-can-now-use-10x-more-item-attributes-to-improve-relevance-of-recommendations/](https://aws.amazon.com/ko/blogs/machine-learning/amazon-personalize-can-now-use-10x-more-item-attributes-to-improve-relevance-of-recommendations/)
* Recommendation Score \(2020.04.06\)
  * [https://aws.amazon.com/ko/about-aws/whats-new/2020/04/amazon-personalize-now-provides-scores-for-recommended-items/](https://aws.amazon.com/ko/about-aws/whats-new/2020/04/amazon-personalize-now-provides-scores-for-recommended-items/)
  * [https://aws.amazon.com/ko/blogs/machine-learning/introducing-recommendation-scores-in-amazon-personalize/](https://aws.amazon.com/ko/blogs/machine-learning/introducing-recommendation-scores-in-amazon-personalize/)
* Recommendation Filters \(2020.06.08\)
  * [https://aws.amazon.com/ko/about-aws/whats-new/2020/06/introducing-recommendation-filters-in-amazon-personalize/](https://aws.amazon.com/ko/about-aws/whats-new/2020/06/introducing-recommendation-filters-in-amazon-personalize/)
  * [https://aws.amazon.com/ko/blogs/machine-learning/introducing-recommendation-filters-in-amazon-personalize/](https://aws.amazon.com/ko/blogs/machine-learning/introducing-recommendation-filters-in-amazon-personalize/)
* Handling missing metadata \(2020.07.02\)
  * [https://aws.amazon.com/ko/about-aws/whats-new/2020/07/amazon-personalize-improved-handling-missing-metadata/](https://aws.amazon.com/ko/about-aws/whats-new/2020/07/amazon-personalize-improved-handling-missing-metadata/)
* User personalization \(2020.08.17\)
  * [https://aws.amazon.com/ko/about-aws/whats-new/2020/08/amazon-personalize-can-now-create-up-to-50-better-recommendations-for-fast-changing-catalogs-of-new-products-and-fresh-content/](https://aws.amazon.com/ko/about-aws/whats-new/2020/08/amazon-personalize-can-now-create-up-to-50-better-recommendations-for-fast-changing-catalogs-of-new-products-and-fresh-content/)
  * [https://aws.amazon.com/ko/blogs/machine-learning/amazon-personalize-can-now-create-up-to-50-better-recommendations-for-fast-changing-catalogs-of-new-products-and-fresh-content/](https://aws.amazon.com/ko/blogs/machine-learning/amazon-personalize-can-now-create-up-to-50-better-recommendations-for-fast-changing-catalogs-of-new-products-and-fresh-content/)
* Incrementally add items and users \(2020.10.02\)
  * [https://aws.amazon.com/ko/blogs/machine-learning/simplify-data-management-with-new-apis-in-amazon-personalize/?nc1=b\_r](https://aws.amazon.com/ko/blogs/machine-learning/simplify-data-management-with-new-apis-in-amazon-personalize/?nc1=b_r)
* Training time improvement \(2020.10.09\)
  * [https://aws.amazon.com/ko/about-aws/whats-new/2020/10/amazon-personalize-announces-improvements-that-reduce-model-training-time-by-up-to-40-percent-and-latency-for-generating-recommendations-by-up-to-30-percent/](https://aws.amazon.com/ko/about-aws/whats-new/2020/10/amazon-personalize-announces-improvements-that-reduce-model-training-time-by-up-to-40-percent-and-latency-for-generating-recommendations-by-up-to-30-percent/)
  * [https://aws.amazon.com/ko/blogs/machine-learning/amazon-personalize-improvements-reduce-model-training-time-by-up-to-40-and-latency-for-generating-recommendations-by-up-to-30/](https://aws.amazon.com/ko/blogs/machine-learning/amazon-personalize-improvements-reduce-model-training-time-by-up-to-40-and-latency-for-generating-recommendations-by-up-to-30/)
* Dynamic Filters for applying Business Rules \(2020.11.13\)
  * [https://aws.amazon.com/ko/about-aws/whats-new/2020/11/apply-business-rules-amazon-personalize-recommendations-on-the-fly/](https://aws.amazon.com/ko/about-aws/whats-new/2020/11/apply-business-rules-amazon-personalize-recommendations-on-the-fly/)
  * [https://aws.amazon.com/ko/blogs/machine-learning/amazon-personalize-now-supports-dynamic-filters-for-applying-business-rules-to-your-recommendations-on-the-fly/](https://aws.amazon.com/ko/blogs/machine-learning/amazon-personalize-now-supports-dynamic-filters-for-applying-business-rules-to-your-recommendations-on-the-fly/)

## 2. Simple Updates

### Contextual Information

User-Personalization 또는 Personalized-Ranking 레시피를 사용하는 경우, user의 현재 위치, 사용중인 기기, 시간&요일 등의 메타 데이터 속성을 기반으로 추천 결과를 필터링할 수 있습니다. 이벤트 발생 시 사용자 환경에서 수집상황별 메타 데이터를 포함하면 기존 user에게 보다 개인화된 추천을 제공할 수 있습니다. 

```python
{
  "name": "DEVICE",
  "type": [
      "string",
      "null"
  ],
  "categorical": true
},
{
  "name": "TIMESTAMP",
  "type": "long"
},
{
  "name": "IMPRESSION",
  "type": "string"
}
```

### Recommendation Score

각 user-item pair$$(u, i)$$에 대한 score는 특정 user와 item 간의 유사도 수준을 아래 수식과 같이 내적으로 표현합니다. \($$\bar{w}_u, w_i$$는 각각 학습된 user 및 item 임베딩 벡터\)

$$
\text{score}(u,i) = \dfrac{\exp(\bar{w}_u^Tw_i)}{\exp(\sum_j\bar{w}_u^Tw_j)}
$$

수식에 의하면 item 셋의 모든 item을 대상으로 계산하기 때문에 점수를 상대적으로 해석해야 합니다. 예를 들어, item이 3개일 경우 score는 0.6, 0.3, 0.1이 될 수 있지만, 1만개일 경우 평균 점수는 1/10,000 이기에, 최고 score를 받은 item도 score가 작을 수 있습니다.

{% hint style="info" %}
SIM와 Popularity-Count 모델은 이 기능을 지원하지 않습니다.
{% endhint %}

```python
get_recommendations_response = personalize_runtime.get_recommendations(
    campaignArn = '[YOUR ARN]',
    userId = str(user_id),
)
item_list = get_recommendations_response['itemList']
item_list[0:2]

-> 
[{'itemId': '5989', 'score': 0.0095232},
 {'itemId': '7147', 'score': 0.0056224}]
```

### Recommendation Filters

* interaction 정보에 대한 필터링과 metadata 정보에 대한 필터링이 모두 가능합니다.
* interaction: [https://aws.amazon.com/ko/blogs/machine-learning/introducing-recommendation-filters-in-amazon-personalize/](https://aws.amazon.com/ko/blogs/machine-learning/introducing-recommendation-filters-in-amazon-personalize/)
  * 한 번 추천을 받고 유저가 선택\(예: 클릭, 다운로드\)한 것에 대해서, 다시 추천을 원하지 않을 경우 유용합니다.
  * `EXCLUDE itemId WHERE INTERACTIONS.event_type in ("Click","Download")`
  * interaction 필터링의 경우, 최근 100개의 실시간 interaction과 최근 200개의 historical interaction만 고려합니다.
* metadata: [https://aws.amazon.com/ko/blogs/machine-learning/enhancing-recommendation-filters-by-filtering-on-item-metadata-with-amazon-personalize/](https://aws.amazon.com/ko/blogs/machine-learning/enhancing-recommendation-filters-by-filtering-on-item-metadata-with-amazon-personalize/)
  * `EXCLUDE ItemId WHERE item.genre in ("Comedy")`

### Null type

스키마 데이터타입 정의 시 null도 허용하게 업데이트됨으로써, 아래와 같은 타입들을 정의할 수 있습니다.

* float / double / int / long / string / boolean / null

### Dynamic Filters

* `IN, =` operator에 dynamic filtering 적용이 가능합니다. 단, `NOT IN, <, >, <=, >=` operator는 여전히 static filter를 사용해야 합니다.
* dollar sign\($\)을 사용하여 placehold 파라메터를 추가하고 상황에 따라 값을 설정해 줍니다.
* 예시: `INCLUDE Item.ID WHERE items.GENRE IN ($GENRE) | EXCLUDE ItemID WHERE item.DESCRIPTION IN ("$DESC”)`
* 참조: [https://docs.aws.amazon.com/personalize/latest/dg/filter-expressions.html](https://docs.aws.amazon.com/personalize/latest/dg/filter-expressions.html)

## 3. Recipes

### Popularity-count

* 모든 user의 행동 데이터를 기반으로 가장 인기 있는 item을 추천합니다. 별도의 하이퍼파라메터 설정이 필요 없으며, baseline 모델로 활용할 수 있습니다.

### User-personalization

* 종래의 HRNN, HRNN-Meta, HRNN-Coldstart를 모두 포괄하며, 내부적으로 MAB\(Multi-Armed Bndits\)을 사용하여 신규 아이템이 자주 업데이트될 때에 적합합니다; [https://docs.aws.amazon.com/personalize/latest/dg/native-recipe-new-item-USER\_PERSONALIZATION.html](https://docs.aws.amazon.com/personalize/latest/dg/native-recipe-new-item-USER_PERSONALIZATION.html)
* MAB에 대한 정보는 아래 블로그를 참조해 주세요.
  * [https://housekdk.gitbook.io/ml/ml/rl](https://housekdk.gitbook.io/ml/ml/rl)
* 주요 하이퍼파라메터
  * `recency_mask (Default = True)`: 이 값을 False로 지정 시, 최신 popularity 트랜드 결과를 반영하지 않습니다. 과거 아이템의 추천 비중을 높이고자 할 때는 False로 설정해 주세요.
  * `exploration_weight (Default = 0.3)`: 이 경우 70%는 interaction dataset에서, 30%는 item dataset에서 추천. 단, exploration\_weight = 1이라고 100% item dataset에서 추천하는 것은 아니고 극히 적은 확률로 interaction dataset을 참조합니다.
  * `exploration_item_age_cut_off (Default = 30)`. interaction dataset의 가장 마지막 interaction 날짜\(timestamp\)를 기준으로 과거 30일 동안의 item을 item metaset에서 탐색합니다.
  * `min_user_history_length_percentile (Default = 0.0, range = [0, 1])`: 솔루션 학습 시 너무 적은 user history를 배제합니다. 이 파라메터는 HPO 튜닝이 불가능합니다.
  * `max_user_history_length_percentile (Default = 0.99, range = [0, 1])`: 솔루션 학습 시 너무 많은 user history를 배제합니다. 이 파라메터는 HPO 튜닝이 불가능합니다.

### SIMS

* Item-based 협업 필터링으로 cosine similiarity로 유사 아이템을 검색합니다. 직관적으로 user-item 상호 작용 정보를 통해 비슷한 user를 찾을 수 있기에, 내가 보지 않았지만 나와 비슷한 user가 봤던 item을 추천하는 것입니다.
* Cold-start problem / sparsity / popularity bias 이슈가 있지만, 충분한 데이터가 확보되었다면 좋은 성능을 보여줍니다.
* 주요 하이퍼파라메터
  * `popularity_discount_factor (Default = 0.5, range = [0, 1])`: 유사도를 계산할 때 popularity와 correlation 사이의 균형을 조절합니다. 0으로 설정 시, popular한 item만 선택합니다.
  * `min_cointeraction_count (Default = 3, range = [0, 10])`: item pair 간의 유사성을 계산하는 데 필요한 최소 interaction 수입니다.
  * `min_user_history_length_percentile (Default = 0.005, range = [0, 1])`: 솔루션 학습 시 너무 적은 user history를 배제합니다. 이 파라메터는 HPO 튜닝이 불가능합니다.
  * `max_user_history_length_percentile (Default = 0.995, range = [0, 1])`: 솔루션 학습 시 너무 많은 user history를 배제합니다. 이 파라메터는 HPO 튜닝이 불가능합니다.
  * `min_item_interaction_count_percentile (Default = 0.01, range = [0, 1])`: 솔루션 학습 시 포함할 최소 item interaction의 백분위수입니다. 이 파라메터는 HPO 튜닝이 불가능합니다.
  * `max_item_interaction_count_percentile (Default = 0.9, range = [0, 1])`: 솔루션 학습 시 포함할 최대 item interaction의 백분위수입니다. 이 파라메터는 HPO 튜닝이 불가능합니다.

### Personalized reranking

* user의 과거 히스토리에 기반하여 개인화된 랭킹을 추천합니다.
* 주요 하이퍼파라메터
  * `recency_mask (Default = True)`: 이 값을 False로 지정 시, 최신 popularity 트랜드 결과를 반영하지 않습니다. 과거 아이템의 추천 비중을 높이고자 할 때는 False로 설정해 주세요.
  * `min_user_history_length_percentile (Default = 0.0, range = [0, 1])`: 솔루션 학습 시 너무 적은 user history를 배제합니다. 이 파라메터는 HPO 튜닝이 불가능합니다.
  * `max_user_history_length_percentile (Default = 0.99, range = [0, 1])`: 솔루션 학습 시 너무 많은 user history를 배제합니다. 이 파라메터는 HPO 튜닝이 불가능합니다.

## 4. Incremental User/Item/interaction

### Adding New items and users

> item, user 추가 기능은 User-personalization 레시피 사용을 권장합니다. User-personalization 레시피를 사용하지 않는 경우, 솔루션 재학습이 필요합니다.

* 신규 아이템 \(PutItems\)
  * User-personalization 레시피를 사용하는 경우 솔루션을 재학습하는 것이 아닌 메타데이터셋을 업데이트하기에 곧바로 반영됩니다. 즉, 캠페인 업데이트 이후 추가된 아이템이 곧바로 추천에 반영됩니다.
  * 필터링의 경우 20분 내에 반영됩니다.
  * 참조: [https://docs.aws.amazon.com/personalize/latest/dg/importing-items.html](https://docs.aws.amazon.com/personalize/latest/dg/importing-items.html)

```python
personalize_events.put_items(
    datasetArn = 'dataset arn',
    items = [{
        'itemId': 'item ID',
        'properties': "{\"propertyName\": \"item data\"}"   
        },
        {
        'itemId': 'item ID',
        'properties': "{\"propertyName\": \"item data\"}"   
        }]
)
```

* 신규 유저 \(PutUsers\)
  * 신규 유저 \(userId가 없는 유저\)의 경우 인기 아이템만 추천됩니다. PutEvents 작업에서 전달한 sessionId를 사용하여 이벤트가 유저와 연결되며, 이벤트 기록들이 계속 누적됩니다.
  * 필터링의 경우 20분 내에 반영됩니다.
  * 참조: [https://docs.aws.amazon.com/personalize/latest/dg/importing-users.html](https://docs.aws.amazon.com/personalize/latest/dg/importing-users.html)

```python
personalize_events.put_users(
    datasetArn = 'dataset arn',
    users = [{
        'userId': 'user ID',
        'properties': "{\"propertyName\": \"user data\"}"   
        },
        {
        'userId': 'user ID',
        'properties': "{\"propertyName\": \"user data\"}"   
        }]
)
```

### Recording Real-time interaction events

User-personalization 레시피 사용시 솔루션\(모델\) 재학습 필요 없이, 2시간마다 백그라운드에서 솔루션 버전을 자동으로 업데이트합니다.

주의: 자동 업데이트는 모델 파라메터를 전면적으로 업데이트하는 것이 아니라, 내부 feature store에 PutEvent를 통해 쌓인 데이터를 interaction dataset에 업데이트하는 것입니다. 따라서, 일정 주기로 전체 재학습을 수행해야 합니다. \(예: 1주일 단위로 재학습\) 재학습 주기를 측정하기 위해 비즈니스 규칙을 활용하거나, 온라인 매트릭을 모니터링하면서 모델 드리프트 시점을 파악할 수 있습니다.

자동 업데이트에 대한 추가 과금은 없으며, 자동 업데이트 조건은 아래와 같습니다.

* latest solution version의 trainingMode == FULL 이고, 신규 item 또는 신규 interactions data가 마지막으로 자동으로 업데이트한 이후에 있을 경우에만 업데이트됩니다. 업데이트 시점은 콘솔 화면의 campaign detail에서 확인 가능합니다.
* 다만, 생성한 솔루션\(모델\)이 2020년 11월 17일 이전이라면 새로 솔루션을 생성해야 하고, trainingMode = FULL로 세팅해야 합니다.
* 만약, 2시간 자동 업데이트 빈도가 적합하지 않은 경우에는\(예: 30분 단위로 업데이트\) trainingMode = UPDATE를 사용하여 신규 솔루션 버전을 생성하고 수동으로 업데이트할 수 있습니다. 단, 수동 업데이트 시에는 추가 비용이 발생합니다.
* [https://docs.aws.amazon.com/personalize/latest/dg/native-recipe-new-item-USER\_PERSONALIZATION.html\#automatic-updates](https://docs.aws.amazon.com/personalize/latest/dg/native-recipe-new-item-USER_PERSONALIZATION.html#automatic-updates)
* Event Tracker 생성

```python
response = personalize.create_event_tracker(
    name='MovieClickTracker',
    datasetGroupArn='arn:aws:personalize:us-west-2:acct-id:dataset-group/MovieClickGroup'
)
```

* PutEvents
  * eventId는 unique event를 판별하기 위한 목적으로 모델에서 활용되지 않으며, eventId를 지정하지 않으면 자동으로 생성해 줍니다. eventType은 필수 값입니다.

```python
personalize_events.put_events(
    trackingId = 'tracking_id',
    userId= 'USER_ID',
    sessionId = 'session_id',
    eventList = [{
        'sentAt': TIMESTAMP,
        'eventType': 'EVENT_TYPE',
        'properties': "{\"itemId\": \"ITEM_ID\"}"
        }]
)

# 여러 이벤트를 동시에 등록하는 것도 가능
personalize_events.put_events(
    trackingId = 'tracking_id',
    userId= 'user555',
    sessionId = 'session1',
    eventList = [{
        'eventId': 'event1',
        'sentAt': '1553631760',
        'eventType': 'like', # like 이벤트
        'properties': json.dumps({
            'itemId': 'choc-panama',
            'eventValue': 4,
            'numRatings': 0    
            })
        }, {
        'eventId': 'event2',
        'sentAt': '1553631782',
        'eventType': 'rating', # rating 이벤트
        'properties': json.dumps({
            'itemId': 'movie_ten',
            'eventValue': 3,
            'numRatings': 13
            })
        }]
)

# User-personalization 레시피에서 impression 데이터도 삽입 가능
# (itemId2&itemId3는 이후 추천 가능성 적어짐) 
personalize_events.put_events(
    trackingId = 'tracking_id',
    userId= 'userId',
    sessionId = 'sessionId',
    eventList = [{
        'eventId': 'event1',
        'eventType': 'rating',
        'sentAt': 1553631760,
        'itemId': 'item id',
        'recommendationId': 'recommendation id',
        'impression': ['itemId1', 'itemId2', 'itemId3']
        }]
)
```

## 5. FAQs

> [https://github.com/aws-samples/amazon-personalize-samples/blob/master/PersonalizeCheatSheet2.0.md](https://github.com/aws-samples/amazon-personalize-samples/blob/master/PersonalizeCheatSheet2.0.md) 을 먼저 확인해 주세요.

### 추천 결과가 너무 최근성만 반영되는데 어떻게 조절할까요?

* User-personalization 레시피를 사용하고 hyperparameter의 `recency_mask = False(Default=True)`로 부여해 주세요. 그리고 필요에 따라 `exploration_weight, exploration_item_age_cut_off` 두 개의 파라메터도 조절합니다.
  * `exploration_weight = 0.3 (Default)` 이 경우 70%는 interaction dataset에서, 30%는 item dataset에서 추천. 단, exploration\_weight = 1이라고 100% item dataset에서 추천하는 것은 아니고 극히 적은 확률로 interaction dataset을 참조합니다.
  * `exploration_item_age_cut_off = 30 (Default)`. interaction dataset의 가장 마지막 interaction 날짜\(timestamp\)를 기준으로 과거 30일 동안의 item을 item metaset에서 탐색합니다.
  * 또한, item dataset의 데이터셋 필드에서 각 아이템에 대한 `CREATION_TIMESTAMP`를 입력하면 더욱 정확한 cold item의 추천이 가능합니다.
    * 예: A상품의 CREATION\_TIMESTAMP: 2021/04/01, B상품의  CREATION\_TIMESTAMP: 2021/03/01일 때 만약 exploration\_item\_age\_cut\_off = 20이고 전체 interaction dataset의 가장 마지막 timestamp가 2021/04/16이면,  3/28 이후에 있는 상품만 추천하게 되기에 A 상품은 추천이 되지 않습니다.
  * 즉, exploration\_weight, exploration\_item\_age\_cut\_off 두 파라메터의 값을 올리면 조금더 old 아이템이 추천이 되는 효과를 가질 수 있습니다.
  * 또한, exploration\_weight, exploration\_item\_age\_cut\_off는 학습과는 상관이 없기에 재학습 필요 없이 캠페인을 여러 개 생성해서 테스트할 수 있습니다.

### 특정 아이템을 너무 많이 추천해 줘요.

* Interaction 데이터가 충분하지 않은 경우, 알고리즘이 popularity\(인기도\)에 대한 가중치를 높이기에 popularity bias 빈도가 늘어납니다. interaction 데이터를 충분히 확보하고 user-personalization 레시피의 하이퍼파라메터를 조절하세요.
* SIMS recipe의 경우, 인기 위주의 유사 상품을 추천하고 item&user metadata를 이용하지 않기에 특정 아이템들 위주로 추천될 수 있습니다.

### user dataset의 컬럼으로 필터링이 가능한가요?

* item, interaction 컬럼만 필터링 가능하며, user의 경우는 CURRENTUSER만 가능합니다. \(user 컬럼 불가능\)
* 성공 예시
  * `INCLUDE ItemID WHERE Items.GENRE IN ("Comedy") IF CURRENTUSER.AGE > 20`
  * `INCLUDE ItemID WHERE Items.GENRE IN ("Comedy") IF CURRENTUSER.GENDER = "F"`
* 실패 예시
  * `EXCLUDE ItemID WHERE Items.GENRE IN (“Comedy”) IF users.AGE > 20` \# CURRENTUSER만 가능하며, user 컬럼으로 필터링 불가능
  * `INCLUDE ItemID WHERE Items.GENRE IN ("Comedy") IF CURRENTUSER.DIV > 20` \# User Meta에 정의되지 않는 경우
  * `INCLUDE ItemID WHERE Items.GENRE IN ("Comedy") IF CURRENTUSER.GENDER > 20` \# 잘못된 조건을 입력하는 경우

### 프로덕션 반영 전 성능 테스트의 모범 사례를 알려 주세요.

* Using A/B testing to measure the efficacy of recommendations generated by Amazon Personalize: [https://aws.amazon.com/ko/blogs/machine-learning/using-a-b-testing-to-measure-the-efficacy-of-recommendations-generated-by-amazon-personalize/](https://aws.amazon.com/ko/blogs/machine-learning/using-a-b-testing-to-measure-the-efficacy-of-recommendations-generated-by-amazon-personalize/)

### 캠페인 숫자의 제한이 5개인데 늘릴 수 있나요?

* Soft Limit이기에 증가 요청 가능합니다. 아래 페이지를 참조해 주세요.
  * [https://ap-northeast-2.console.aws.amazon.com/servicequotas/home?region=ap-northeast-2\#!/services/personalize/quotas](https://ap-northeast-2.console.aws.amazon.com/servicequotas/home?region=ap-northeast-2#!/services/personalize/quotas)
* Limit 참
  * [https://docs.aws.amazon.com/ko\_kr/personalize/latest/dg/limits.html\#limits-table](https://docs.aws.amazon.com/ko_kr/personalize/latest/dg/limits.html#limits-table)

### Latency가 너무 높습니다.

* item 수가 너무 많을 경우\(예: 2백만 개\) 주로 발생합니다. 학습 시 item 개수는 75만개가 limit이며, 캠페인 배포 시에 item 개수에 따른 제한을 걸어 놓지는 않았지만, item 개수가 지나치게 많은 경우 모델 컨테이너에서 계산을 수행하는 데 시간이 걸립니다. 따라서, 가급적 75만개 미만으로 유지하는 것을 권장합니다. 또한, 저 적은 TPS로 부하 테스트를 시작하고 원하는 TPS로 부하를 늘려서 테스트해 보세요. 아래 limit을 같이 참조 부탁드립니다.
  * Maximum number of items that are considered by a model during training. 750 thousand
  * Maximum number of interactions that are considered by a model during training. 500 million
  * Maximum number of GetRecommendations requests per second per campaign. 500/sec

## References

* Amazon Personalize Deep Dive Series: [https://www.youtube.com/playlist?list=PLhr1KZpdzukd9GSGRy329wahNO\_8TkRo\_](https://www.youtube.com/playlist?list=PLhr1KZpdzukd9GSGRy329wahNO_8TkRo_)

