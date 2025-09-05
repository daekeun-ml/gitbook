# Horangi Korean LLM Leaderboard

## 1. 핵심 평가 체계

***

### **1. 이중 평가 구조**

#### **언어 이해 (Language Understanding)**

* 일문일답식 평가 체계를 사용하므로 입력된 내용을 정확하게 이해하고, 요구된 형식으로 답변하는 능력을 평가
* 객관적이고 정량적인 측정 방식

#### **언어 생성 (Language Generation)**

* 자유 형식으로 모델에게 답변을 출력시키고, GPT-4를 사용한 정성적 평가를 수행
* 작문(writing), 추론(reasoning), 정보 추출(extraction) 등의 평가 축에서 모델의 강점을 검증

## **2. 기술적 프레임워크**

***

* **MT-Bench 프레임워크 활용:** Stability AI사와의 협력하에 이 회사가 개발한 MT-Bench 프레임워크를 활용하여 체계적인 평가 수행
* **멀티턴 대화 평가:** Q\&A 형식의 언어이해와 멀티턴 대화를 통한 생성 능력을 종합적으로 평가하여 실제 사용 환경에 가까운 평가 제공.
* **평가 데이터셋 다양화:** KoBBQ, Korean Hate Speech, AI HUB의 텍스트 윤리검증 데이터 등 공개 데이터셋을 최대한 활용해 평가의 객관성과 신뢰도를 높임
* **한국 문화적 특성 반영:** 오픈소스 언어모델 연구팀 'HAERAE'의 HAERAE\_BENCH\_V1, KMMLU와 'NAVER AI LAB'의 KoBBQ를 활용하여 한국 문화와 사회적 맥락을 고려한 출력 평가
* **제약 사항**: 대상 모델이 대화형 프롬프트에 대해 적절한 응답을 반환하는 것을 전제로 하고 있음. 이에 인스트럭션 튜닝을 통해 이른바 챗봇 능력을 획득하지 못한 모델에 대한 평가는 부적절함.

## 3. Benchmark Dataset

***

### llm-kr-eval

* [https://github.com/wandb/llm-kr-eval/blob/main/DATASET.md](https://github.com/wandb/llm-kr-eval/blob/main/DATASET.md)
* llm-jp-eval를 한국어 평가용으로 수정하고 기존에 공개된 벤치마크 데이터셋을 파인튜닝과 테스트에 알맞게 전처리 (Apache-2.0 라이센스)
* 데이터셋
  * NLI (Natural Language Inference): KorNLI(exact), KoBEST\_HellaSwag(exact), KoBEST\_COPA(exact)
  * QA (Question Answering): KoBEST\_WiC(exact), KMMLU(exact
  * RC (Reading Comprehension): KorSTS(person, spearman), KoBEST\_SN(exact)
  * EL (Entity Linking) : KLUE-NER(set\_f1), KLUE-RE(exact)
  * FA (Fundamental Analysis): Korean-CommonGen(bleu)
* 모델의 순수 능력을 보기 위해서 제로샷 평가
*   포맷

    * 평가

    ```jsx
    {
        "instruction": "전제와 가설의 관계를 entailment, contradiction, neutral 중에서 답변하십시오. 그 외에는 아무것도 포함하지 않는 것을 엄수하십시오. \\n\\n제약:\\n- 전제가 참일 때 가설이 참이면 entailment 출력\\n- 전제가 참일 때 가설이 거짓이면 contradiction 출력\\n- 어느 쪽도 아닌 경우는 neutral 출력.",
        "output_length": 13,
        "metrics": [
            "exact_match"
        ],
        "few_shots": [
             {
                "input": "전제:이 특별한 경우 구매자는 도시에서 출하되기보다는 물건을 구입하기 위해 라스베가스로 오지만 경제는 동일합니다.\\n가설:구매자들은 라스베이거스에서 직접 물건을 구입했다.",
                "output": "entailment"
            },
            {
                "input": "전제:닫힌 보도 위에 손수레를 든 두 남자가 표지판으로 표시되어 있다.\\n가설:두 남자가 닫힌 보도를 수리하고 있다.",
                "output": "neutral"
            },
            ...
        ]
            "samples": [
             {
                "input": "전제:그리고 그가 말했다, \\"엄마, 저 왔어요.\\"\\n가설:그는 학교 버스가 그를 내려주자마자 엄마에게 전화를 걸었다.",
                "output": "neutral"
            },
            {
                "input": "전제:그리고 그가 말했다, \\"엄마, 저 왔어요.\\"\\n가설:그는 한마디도 하지 않았다.",
                "output": "contradiction"
            },
            ...
        ]
    }
    ```

    * 튜닝

    ```jsx
    [
         {
            "ID": "kornli-0",
            "instruction": "전제와 가설의 관계를 entailment, contradiction, neutral 중에서 답변하십시오. 그 외에는 아무것도 포함하지 않는 것을 엄수하십시오. \\n\\n제약:\\n- 전제가 참일 때 가설이 참이면 entailment 출력\\n- 전제가 참일 때 가설이 거짓이면 contradiction 출력\\n- 어느 쪽도 아닌 경우는 neutral 출력.",
            "input": "전제:그리고 그가 말했다, \\"엄마, 저 왔어요.\\"\\n가설:그는 학교 버스가 그를 내려주자마자 엄마에게 전화를 걸었다.",
            "output": "neutral",
            "text": "다음은 작업을 설명하는 지침과 컨텍스트 입력의 조합입니다. 요구를 적절하게 만족시키는 응답을 적으십시오.\\n\\n### 지시:\\n전제와 가설의 관계를 entailment, contradiction, neutral 중에서 답변하십시오. 그 외에는 아무것도 포함하지 않는 것을 엄수하십시오. \\n\\n제약:\\n- 전제가 참일 때 가설이 참이면 entailment 출력\\n- 전제가 참일 때 가설이 거짓이면 contradiction 출력\\n- 어느 쪽도 아닌 경우는 neutral 출력.\\n\\n### 입력:\\n전제:그리고 그가 말했다, \\"엄마, 저 왔어요.\\"\\n가설:그는 학교 버스가 그를 내려주자마자 엄마에게 전화를 걸었다.\\n\\n### 응답:\\nneutral"
        },
        {
            "ID": "kornli-1",
            "instruction": "전제와 가설의 관계를 entailment, contradiction, neutral 중에서 답변하십시오. 그 외에는 아무것도 포함하지 않는 것을 엄수하십시오. \\n\\n제약:\\n- 전제가 참일 때 가설이 참이면 entailment 출력\\n- 전제가 참일 때 가설이 거짓이면 contradiction 출력\\n- 어느 쪽도 아닌 경우는 neutral 출력.",
            "input": "전제:그리고 그가 말했다, \\"엄마, 저 왔어요.\\"\\n가설:그는 한마디도 하지 않았다.",
            "output": "contradiction",
            "text": "다음은 작업을 설명하는 지침과 컨텍스트 입력의 조합입니다. 요구를 적절하게 만족시키는 응답을 적으십시오.\\n\\n### 지시:\\n전제와 가설의 관계를 entailment, contradiction, neutral 중에서 답변하십시오. 그 외에는 아무것도 포함하지 않는 것을 엄수하십시오. \\n\\n제약:\\n- 전제가 참일 때 가설이 참이면 entailment 출력\\n- 전제가 참일 때 가설이 거짓이면 contradiction 출력\\n- 어느 쪽도 아닌 경우는 neutral 출력.\\n\\n### 입력:\\n전제:그리고 그가 말했다, \\"엄마, 저 왔어요.\\"\\n가설:그는 한마디도 하지 않았다.\\n\\n### 응답:\\ncontradiction"
        },
        ...
    ]
    ```

### MT-Bench

* 멀티턴 질의응답 문항들로 이루어진 MT Bench를 한국어로 번역
  * Paper: [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)
  * GitHub: [https://github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat)
  * KoMT-Bench: [https://github.com/LG-AI-EXAONE/KoMT-Bench?tab=readme-ov-file](https://github.com/LG-AI-EXAONE/KoMT-Bench?tab=readme-ov-file)

#### Examples

<table data-header-hidden><thead><tr><th width="139.453125">Category</th><th>MT-Bench</th><th>KoMT-Bench</th></tr></thead><tbody><tr><td><strong>Writing</strong></td><td></td><td></td></tr><tr><td>1st Turn</td><td>Imagine you are writing a blog post comparing two popular smartphone models. Develop an outline for the blog post, including key points and subheadings to effectively compare and contrast the features, performance, and user experience of the two models. Please answer in fewer than 200 words.</td><td>두 개의 인기 스마트폰 모델을 비교하는 블로그 게시물을 작성한다고 가정합니다. 두 모델의 기능, 성능, 사용자 경험을 효과적으로 비교하고 대조할 수 있도록 핵심 사항과 소제목을 포함하여 블로그 게시물의 개요를 작성하세요. 200자 이내로 답하십시오.</td></tr><tr><td>2nd Turn</td><td>Take your previous response and rephrase it as a limerick.</td><td>이전 답변을 충청도 사투리로 재작성하십시오.</td></tr><tr><td><strong>Math</strong></td><td></td><td></td></tr><tr><td>1st Turn</td><td>When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?</td><td>어떤 숫자를 10으로 나눈 나머지는 4입니다. 그 숫자의 두 배를 4로 나눈 나머지를 구하세요.</td></tr><tr><td>2nd Turn</td><td>What about when twice the number is divided by 5?</td><td>그 숫자의 두 배를 5로 나누면 어떨까요?</td></tr><tr><td><strong>Humanities</strong></td><td></td><td></td></tr><tr><td>1st Turn</td><td>Provide insights into the correlation between economic indicators such as GDP, inflation, and unemployment rates. Explain how fiscal and monetary policies affect those indicators.</td><td>GDP, 인플레이션, 실업률과 같은 경제 지표 간의 상관관계에 대한 통찰을 제시하세요. 이러한 지표들에 재정 및 통화 정책이 어떤 영향을 미치는지 설명하세요.</td></tr><tr><td>2nd Turn</td><td>Now, explain them again like I'm five.</td><td>이제 제가 5살이라 생각하고 다시 설명해 주세요.</td></tr></tbody></table>

