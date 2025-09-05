---
description: 'Content Level: 200-300'
---

# Common Strategies to Consider When Generating Synthetic Data

## Suggested Pre-Reading

* [The Necessity of Synthetic Data: Core Requirements for Modern AI Development](the-necessity-of-synthetic-data.md)
* [Seed Data-Based Synthetic Data Generation Approach (Persona-Specific)](seed-data-based-synthetic-data-generation-approach.md)
* [Seedless Synthetic Data Generation Approach (Seedless Methods)](seedless-synthetic-data-generation-approach.md)

## TL;DR

합성 데이터를 만들어 모델을 학습시키는 과정에서는, 접근 방식(seed vs seedless)에 관계없이 **공통적으로 중요하게 신경 써야 할 요소들**이 있습니다. 이러한 요소들은 데이터의 **품질, 안전성, 유용성**을 담보하고, 합성 데이터 활용을 지속적으로 개선하는 데 필수적입니다. 본 섹션에서는 다음 다섯 가지를 중점적으로 살펴보겠습니다: (1) 추가적인 데이터 증강 기법 (예: _Evolve-Instruct_), (2) PII 제거, (3) 데이터 신뢰성 검증 (LLM 평가자 활용), (4) 레드팀을 통한 검증 (Responsible AI), (5) 합성 데이터의 버전 관리 전략.

## 1. 합성 데이터 추가 증강 – WizardLM의 Evolve Instruct 기법

***

초기에 생성한 합성 데이터셋을 그대로 사용하는 것도 좋지만, **데이터를 더 풍부하게 증강(augmentation)**&#xD558;여 모델 훈련 효과를 극대화할 수 있습니다. [WizardLM 논문](https://arxiv.org/abs/2304.12244)에서는 _Evolve-Instruc&#x74;_&#xB77C;는 기법을 도입하여, **기존 프롬프트를 점진적으로 변형해 난이도와 다양성을 높인 새로운 프롬프트들을 생성**했습니다. 간단히 말해, 기본 질문을 “한 단계 진화”시켜 더 복잡한 질문으로 만들고 이를 데이터셋에 추가하는 것입니다.

Evolve-Instruct 기법의 주요 아이디어는 아래와 같습니다:

* **In-Depth Evolving (심화 방향 진화):** 한 프롬프트에 대해 _추가 제약이나 요구사항을 부&#xC5EC;_&#xD558;거나, _질문의 범위를 깊게 또는 넓게_ 만들어 난이도를 높입니다. 예를 들어 원래 프롬프트가 “기후 변화가 농업에 미치는 영향은?”이었다면, 심화 진화된 프롬프트는 “기후 변화가 **식량 안보와 지역사회 경제를 포함하여** 농업에 미치는 복합적 영향은? 구체적 사례를 들어 설명하라.”처럼 변형될 수 있습니다. WizardLM 논문에서는 이를 위해 LLM에게 다음과 같은 역할을 부여했습니다: _“당신의 목적은 주어진 프롬프트를 사람이 이해할 수 있으면서도 더 복잡한 버전으로 재작성하는 것이다. 프롬프트에 새로운 제약이나 추가 질문을 넣거나, 더 많은 단계의 추론이 필요하도록 만들어라.”_. 이 지시에 따라 모델은 자동으로 기존 질문을 한층 어려운 질문으로 바꾸어 줍니다.
* **In-Breadth Evolving (폭 확장 방향 진화):** 현재 데이터셋에 없는 _새로운 주제나 기&#xC220;_&#xC744; 프롬프트로 추가하여 **토픽 커버리지를 넓히는** 방법입니다. 예를 들어 원 데이터셋이 주로 수학 문제로 이루어져 있었다면, 폭 확장 단계를 통해 과학, 역사, 철학 관련 질문들도 생성해 추가하는 식입니다. 이를 통해 데이터셋 전체로 보면 훨씬 폭넓은 범위의 지식을 포함하게 됩니다.

WizardLM은 이러한 Evol-Instruct로 **약 70,000개의 다양한 난이도의 지시 데이터**를 생성했고, 이를 7억(7B) 파라미터 모델에 파인튜닝하여 **ChatGPT 수준에 버금가는** 성능을 이끌었다고 보고했습니다. 이는 작은 모델도 **질적으로 우수한 합성 데이터**만 있으면 큰 모델을 따라잡을 만큼 향상될 수 있음을 보여줍니다. 실제로 WizardLM-7B 모델은 수학 문제 해결이나 코드 작성 등 복잡한 작업에서 훨씬 큰 모델들에 필적하는 결과를 보였습니다.

우리도 합성 데이터를 만들 때, **Evolve-Instruct 개념을 응용**할 수 있습니다. 예를 들어 이미 생성한 Q\&A나 지시문 세트가 있다면, 그중 일부를 변형해서 더 어려운 버전, 더 구체적인 버전, 혹은 아예 다른 맥락을 추가한 버전으로 늘릴 수 있습니다. 이렇게 하면 모델이 같은 주제를 여러 각도에서 다루도록 훈련되어 **일반화 성능**이 향상됩니다. 다만, 자동 변형 과정에서 **의미가 통째로 바뀌거나 비합리적인 요구사항이 들어가지 않도록** 주의해야 합니다. WizardLM에서는 사람이 직접 하나하나 수정한 것이 아니라 LLM이 변형했지만, 그때도 “말이 되도록, 인간이 이해할 수 있도록” 등의 조건을 달아 **품질을 통제**했습니다. 우리도 마찬가지로, 증강된 프롬프트/응답을 한번 더 LLM이나 휴먼으로 검증해보는 것이 좋습니다.

요약하면, _Evolve-Instruc&#x74;_&#xB294; **기존 합성 데이터의 양과 난이도를 한층 끌어올리는 자동화된 증강 방법**이며, 이를 통해 모델의 능력을 전반적으로 향상시킬 수 있습니다. 특히 모델에게 **쉽지 않은 질문들까지 훈련**시키려면 사람이 일일이 어려운 질문을 제작하기 어렵기 때문에, 이러한 LLM 기반 자동 난이도 상승 기법이 큰 도움이 됩니다.

코드 스니펫은 아래와 같습니다. 자세한 구현은 [이 GitHub 리포지토리](https://github.com/daekeun-ml/synthetic-qa-generation/tree/main/evolve-instruct)을 참조하세요.

```python
import random
from typing import List, Dict

class EvolveInstructAugmenter:
    def __init__(self):
        self.evolution_methods = [
            'add_constraints',
            'deepen_complexity', 
            'increase_reasoning',
            'add_breadth',
            'complicate_input'
        ]
    
    def evolve_instruction(self, original_instruction: str) -> str:
        """지시사항 진화"""
        method = random.choice(self.evolution_methods)
        
        evolution_prompts = {
            'add_constraints': f"Add one constraint to make this more complex:\n{original_instruction}",
            'deepen_complexity': f"Increase depth and complexity:\n{original_instruction}",
            'increase_reasoning': f"Require more reasoning steps:\n{original_instruction}",
            'add_breadth': f"Broaden the scope:\n{original_instruction}",
            'complicate_input': f"Make the input more complex:\n{original_instruction}"
        }
        
        return evolution_prompts[method]
```

## 2. PII 제거 – 개인 정보 필터링

***

합성 데이터를 생성하거나 가공할 때 **개인식별정보(PII)**&#xAC00; 포함되지 않도록 관리하는 것은 매우 중요합니다. PII란 이름, 주소, 주민번호, 전화번호, 이메일 등 개인을 식별할 수 있는 정보를 말하며, 실제 사용자 데이터를 다루는 경우 이러한 정보가 의도치 않게 합성 결과에 들어갈 수 있습니다. 또한 웹 크롤링 등으로 얻은 원천 데이터에는 PII가 섞여 있을 수 있어, 이를 제거하지 않고 모델을 훈련하면 **프라이버시 침해**나 **규정 위반** 문제가 발생할 수 있습니다.

이를 해결하기 위해 **자동 PII 식별/마스킹 도구**를 활용하는 것이 권장됩니다. AWS에서는 _Amazon Comprehen&#x64;_&#xB77C;는 NLP 서비스를 통해 **텍스트 내 다양한 PII 항목을 탐지 및 제거**할 수 있습니다. Comprehend의 PII 검출 기능은 수십 가지 유형의 민감 정보를 인식하며 (예: 이름, 신용카드 번호, 주소, 은행 계좌 등), 각 발견된 항목에 대해 **신뢰도 점수**와 **위치 정보**까지 제공합니다. 이를 활용하면 우리가 생성한 합성 데이터에서 PII로 의심되는 문자열을 자동으로 **마스킹**하거나 삭제할 수 있고, 어떤 항목이 제거됐는지 로그를 남겨서 **추적 관리**도 가능해집니다. Comprehend는 ML 기반으로 문맥을 이해해 PII를 찾기 때문에, 정규식이나 키워드 매칭만으로 찾기 어려운 패턴도 인지합니다. 예를 들어 “우리 엄마 번호는 010-1234-5678이에요” 같은 문장에서 숫자 패턴만 보고 식별하는 게 아니라, 주변 문맥으로 이것이 전화번호임을 파악하는 식입니다.

**실용적인 적용:** 합성 데이터를 생성할 때, 가급적 **애초에 PII가 포함된 원본**을 사용하지 않는 것이 좋습니다. 그러나 부득이하게 실제 사용자 로그나 대화 등을 seed로 쓸 때는, 먼저 Comprehend 등을 돌려 **원천 데이터를 필터링**합니다. 혹시 합성 과정에서 모델이 임의의 번호나 이메일 등을 생성하는 경우 (LLM이 가짜로 만들기도 함), 그 결과물에도 PII 검출을 적용해 **후처리 단계**를 거칩니다. AWS의 경우 SageMaker 데이터 처리 파이프라인에 Comprehend를 통합하여, 데이터 준비 단계에서 자동으로 PII를 걸러내는 워크플로우를 구현할 수 있습니다. 예컨대 SageMaker Data Wrangler의 흐름에 Comprehend PII 검출 노드를 삽입해, CSV나 JSON 데이터 속 민감 정보를 마스킹(예: “”) 처리해버리는 식입니다. 이렇게 하면 이후 단계에서는 안심하고 데이터를 쓸 수 있습니다.

**주의점:** PII 제거 시에는 **과도하거나 불필요한 정보까지 제거하지 않도록** 균형을 잡아야 합니다. 예를 들어 “마이크로소프트 회사는…”에서 “마이크로소프트”를 개인 이름으로 잘못 인식해 지우면 문장 의미가 훼손됩니다. Comprehend는 컨텍스트를 보긴 하지만, 완벽하지 않을 수 있으므로 결과를 한번 검토하는 것이 좋습니다. 또한 합성 데이터에는 원래부터 특정인에 대한 정보가 들어가면 안 되므로 (사실적이든 허구든), **가능한 한 일반화**하고 **익명화된 컨텐츠**로 생성하도록 유도하는 것이 중요합니다.

결론적으로, 합성 데이터 파이프라인에 **PII 필터를 내장**하여 개인정보가 포함될 위험을 줄이고, 프라이버시 및 규제 요구사항을 준수하는 것이 바람직합니다. AWS Comprehend와 Amazon SageMaker를 활용하면 이러한 **자동화된 PII 레드액션(redaction)** 프로세스를 손쉽게 구축하고 확장성 있게 운영할 수 있습니다.

아이디어를 구체화하기 위한 코드 스니펫은 아래와 같습니다.

```python
import boto3
import json

class PIIRemover:
    def __init__(self):
        self.comprehend = boto3.client('comprehend')
        
    def detect_and_remove_pii(self, text: str) -> Dict:
        """PII 탐지 및 제거"""
        try:
            response = self.comprehend.detect_pii_entities(
                Text=text,
                LanguageCode='en'
            )
            
            entities = response['Entities']
            cleaned_text = self._mask_pii_entities(text, entities)
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'pii_entities': entities,
                'is_safe': len(entities) == 0
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'original_text': text,
                'cleaned_text': text,
                'is_safe': False
            }
    
    def _mask_pii_entities(self, text: str, entities: List[Dict]) -> str:
        """PII 엔티티 마스킹"""
        # 뒤에서부터 처리하여 인덱스 변화 방지
        sorted_entities = sorted(entities, key=lambda x: x['BeginOffset'], reverse=True)
        
        masked_text = text
        for entity in sorted_entities:
            start = entity['BeginOffset']
            end = entity['EndOffset']
            entity_type = entity['Type']
            
            # 엔티티 타입에 따른 마스킹
            mask = f"<{entity_type}>"
            masked_text = masked_text[:start] + mask + masked_text[end:]
        
        return masked_text
```

## 3. 데이터 신뢰성 검증 – LLM-as-a-Judge

***

합성 데이터의 품질을 확보하기 위해서는 **생성된 데이터에 대한 검증 단계**가 필요합니다. 하지만 대량의 합성 데이터를 일일이 사람이 검사하기는 현실적으로 어렵습니다. 이때 **LLM을 평가자(Judge)로 활용**하는 기법이 유용합니다. _LLM-as-a-Judg&#x65;_&#xB780; 한마디로, **한 모델(평가자)이 다른 모델(생성자)의 출력 품질을 평가**하는 구조를 말합니다. 예를 들어 우리가 합성한 질문-답변 쌍에 대해, ChatGPT나 GPT-4 같은 상위 모델에게 “이 질문에 대한 답이 정확하고 완결적인지 채점해줘”라고 묻는 것입니다.

LLM 평가자는 다양한 측면에서 데이터를 점검할 수 있습니다:

* **정확성(Accuracy):** 답변이 질문에 맞는 올바른 내용을 담고 있는지. 사실관계 오류나 논리 오류가 없는지 확인합니다. 예: _“질문: 태양은 어느 행성 주위를 도는가? 답: 화성.”_ (틀린 답변을 잡아냄).
* **완전성(Completeness):** 답변이 질문에 요구된 정보를 모두 포함하는지. 불충분한 답변이면 지적합니다.
* **관련성(Relevance):** 생성된 문장이 문맥에 맞는지, 질문과 상관없는 내용이 섞이지 않았는지 평가합니다.
* **유창성 및 일관성:** 문법이나 어투가 이상하지 않은지, 전반적으로 자연스러운지 봅니다. 합성 데이터는 가끔 어색한 표현이 있을 수 있는데, 평가 LLM이 그런 문장을 잡아낼 수 있습니다.
* **유해성 여부:** 답변 내용에 유해하거나 편향된 요소가 없는지도 점검 가능합니다. (이 부분은 레드팀과도 겹치지만, 데이터 단계에서 1차로 걸러낼 수 있습니다.)

LLM-as-a-Judge를 구현하는 방법은 크게 두 가지입니다:

1. **프롬프트 기반 평가 (In-Context Learning):** 별도 튜닝 없이 GPT-4 등의 API에 평가 기준을 프롬프트로 제시해 바로 사용하는 것입니다. 예컨대 시스템 프롬프트에 “You are a strict grader. Check the following Q\&A…” 식으로 역할을 설정하고, 사용자 메시지에 합성 Q\&A와 평가항목을 주면, 모델이 “점수: 7/10, 이유: \~\~\~” 또는 “올바름 여부: false, 이유: \~\~\~” 같은 평가를 내리게 할 수 있습니다.
2. **평가자 모델 파인튜닝:** 아예 평가만 전문적으로 하도록 **별도의 모델을 훈련**시키는 것입니다. 최근 여러 연구가 GPT-4 출력 데이터를 활용해 _스타일 평가, 사실 평가_ 등에 특화된 **보조 모델**을 만들고 있습니다. 이러한 모델은 속도가 빠르고 비용이 낮아 대량 데이터 검증에 적합합니다. 다만 처음에는 기본 LLM으로 채점한 결과를 학습시켜야 하기 때문에 어느 정도 시간과 노력이 듭니다.

어느 방식을 쓰든, 핵심은 **명확한 평가 기준**을 설정하는 것입니다. LLM은 사람과 달리 암묵적인 판단을 하기 때문에, 프롬프트로 “정확성과 완전성만 보고 이진 판단해” 혹은 “각 답변에 1\~5점 채점해” 등 구체적으로 지시해야 일관된 결과를 얻을 수 있습니다. 예를 들어 “질문과 답을 보고, 답이 질문에 맞게 잘 답했으면 ‘정답’, 틀렸거나 불충분하면 ‘오답’이라고만 답하라”는 식으로 단순 레이블링을 시키면 비교적 정확히 검출해냅니다. 또는 두 개 이상의 모델 답변을 나란히 주고 “어느 쪽이 더 나은지 골라”라고 하여 **랭킹 평가(pairwise comparison)**&#xB97C; 할 수도 있습니다. 이는 주관적 품질 평가에 강점이 있습니다 (안내 문구의 친절함 등은 상대 비교가 정확도가 높습니다).

AWS 환경에서는 이러한 LLM 평가를 자동화하는 솔루션으로 **Bedrock**을 활용할 수 있습니다. Bedrock을 통해 GPT-4나 Anthropic Claude 같은 모델에 접근하여 대량의 프롬프트 평가를 병렬 수행하는 파이프라인을 만들 수 있습니다. 또한 _AWS FMEval_ 같은 도구는 미리 정의된 평가 템플릿에 따라 모델 응답을 평가하는 기능을 제공해, 개발자가 일일이 프롬프트를 작성하지 않고도 **표준화된 평가 루틴**을 사용할 수 있게 합니다. (Data Reply 사의 예에서는 Giskard, LangFuse 같은 오픈소스 툴과 AWS FMEval을 결합해 평가/테스트 환경을 구축했다고 합니다.)

결국 **LLM-as-a-Judge**를 도입하면, 합성 데이터의 _자동 품질 통&#xC81C;_&#xAC00; 가능해집니다. 사람이 직접 만들었다면 바로 믿겠지만, 모델이 만든 데이터는 잘못된 경우가 분명 존재하므로 반드시 이런 **검수 단계**를 거쳐야 합니다. 실제 OpenAI도 GPT-4로 생성한 데이터로 GPT-3.5를 학습시킬 때, **다시 GPT-4로 결과를 검토**하는 식의 루프를 활용했다고 알려져 있습니다. 이러한 _자기평가(self-evaluation) 및 교차평&#xAC00;_&#xB97C; 수행하면, 잘못된 합성 데이터(정답이 틀린 QA 등)를 걸러내고 **신뢰도 높은 데이터셋**을 확보할 수 있습니다.

아이디어를 구체화하기 위한 코드 스니펫은 아래와 같습니다.

```python
import boto3
import json

class DataQualityJudge:
    def __init__(self):
        self.bedrock = boto3.client('bedrock-runtime')
        
    def evaluate_data_quality(self, instruction: str, response: str) -> Dict:
        """데이터 품질 평가"""
        evaluation_prompt = f"""
Evaluate the quality of this instruction-response pair on a scale of 1-10:

Instruction: {instruction}
Response: {response}

Rate on these criteria:
1. Relevance (1-10): How well does the response address the instruction?
2. Accuracy (1-10): Is the information factually correct?
3. Completeness (1-10): Does the response fully answer the instruction?
4. Clarity (1-10): Is the response clear and well-structured?
5. Safety (1-10): Is the content safe and appropriate?

Provide scores in JSON format:
{{"relevance": X, "accuracy": X, "completeness": X, "clarity": X, "safety": X, "overall": X, "feedback": "brief explanation"}}
"""
        
        try:
            response = self.bedrock.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 500,
                    'messages': [{'role': 'user', 'content': evaluation_prompt}]
                })
            )
            
            result = json.loads(response['body'].read())
            evaluation_text = result['content'][0]['text']
            
            # JSON 추출
            import re
            json_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                return scores
            
            return {'error': 'Could not parse evaluation'}
            
        except Exception as e:
            return {'error': str(e)}
```

## 4. 책임감 있는 AI를 위한 Red Teaming

***

합성 데이터로 모델을 훈련시킨 후 (혹은 훈련 중에) 반드시 점검해야 할 부분이 바로 **모델의 안전성과 악용 가능성**입니다. 레드팀(Red teaming)이란 _모델의 취약점을 찾기 위해 의도적으로 공격적 시나리오를 테스&#xD2B8;_&#xD558;는 것을 말합니다. 이는 모델 개발자 입장에서 일종의 모의 해킹이나 스트레스 테스트로 볼 수 있습니다. 구체적으로, 레드팀은 **모델이 부적절한 응답을 생성하거나, 외부 조작에 취약한지** 등을 검증합니다.

합성 데이터를 통해 모델을 만들었다면, 실제 사용자 데이터로 훈련한 경우보다 **잠재적인 위험**을 더 면밀히 살펴봐야 합니다. 왜냐하면 합성 데이터에는 현실 세계의 미묘한 편향이나 금기사항이 충분히 반영되지 않았을 수 있기 때문입니다. 따라서 레드팀 절차를 도입해 다음과 같은 관점들을 점검합니다:

* **할루시네이션(Hallucination):** 모델이 자신있게 사실이 아닌 정보를 만들어내는 경향. 합성 데이터로 학습한 모델은 특정 패턴에 과적합되어 사실 확인 없이도 답변할 수 있으므로, 사실 확인 질문 등에 대해 엉뚱한 답을 하지 않는지 테스트합니다.
* **유해 발언 및 편향:** 모델이 인종차별적, 성차별적, 폭력적, 음란한 등의 콘텐츠를 생성하는지 시나리오별로 검사합니다. 합성 데이터에 이런 요소가 없더라도, 사전 학습에서 내재된 내용이 나올 수 있어 _민감 주제 프롬프&#xD2B8;_&#xB97C; 다양하게 시도해봅니다.
* **프롬프트 조작 취약성:** 예를 들어 “**시스템**” 메시지를 무시하게 하는 prompt injection 공격이나, 모델에게 정책 위반 답변을 유도하는 기법에 취약한지 시험합니다. 합성 데이터로 RLHF 과정을 대체했다면 안전 장치가 부족할 수 있으므로 특히 중요한 부분입니다.
* **정보 유출:** 모델이 훈련 데이터(합성 데이터 포함) 내용을 그대로 반복하거나, 개인 정보 등을 드러내지 않는지 확인합니다.

AWS 상에서 이러한 레드팀 활동을 지원하는 프레임워크도 존재합니다. 예를 들어 **Amazon SageMaker Clarify**는 데이터 및 모델의 편향성, 특이 사례를 분석하는 도구로, 훈련 데이터의 _공정성(Fairness)_ 분석에 활용할 수 있습니다. 하지만 Clarify는 주로 정량적 분석에 가깝고, 레드팀은 보다 **동적이고 창의적인 공격 시나리오**까지 포함합니다. AWS와 Data Reply 등이 제공하는 **Red Teaming 서비스/가이드**에서는, 오픈소스 툴 (예: _Giskard_ – 모델 테스트, _LangFuse_ – 모니터링 등)과 AWS의 인프라를 결합해 **자동화된 레드팀 플랜**을 구축하는 방법을 소개합니다. 이를 통해 개발자는 사전에 정의된 위험 시나리오 스위트에 따라 모델을 평가하고, 문제가 발견되면 데이터나 모델 수정을 피드백 루프로 돌릴 수 있습니다.

예를 들어, 레드팀 테스트 시나리오로 “증오 발언 생성 시도”, “비밀 정보 추출 시도”, “모델 정책 우회 시도” 등의 프롬프트 모음을 준비해 둡니다. 그런 다음 Bedrock이나 자체 배포한 LLM API를 통해 이 프롬프트들을 모델에 입력하고 응답을 수집합니다. 결과를 Giskard 같은 툴로 자동분석하여, 응답 중 위험도가 높은 것을 플래그하거나 사람 검토가 필요한 케이스를 뽑아냅니다. 이러한 작업을 정기적으로, 또는 모델 업데이트 때마다 수행하면 **모델의 이상 행동을 조기에 발견**할 수 있습니다. 특히 합성 데이터로 학습된 모델은 특정 분포에 치우쳐 있어, 예기치 못한 입력에 더 취약할 수 있으므로, 레드팀을 통해 **배포 전에 충분히 검증**해야 합니다.

AWS Bedrock Guardrails를 연동한 코드 스니펫은 아래와 같습니다.

```python
class RedTeamingValidator:
    def __init__(self):
        self.bedrock = boto3.client('bedrock-runtime')
        self.guardrails_id = "your-guardrails-id"
        
    def red_team_evaluation(self, content: str) -> Dict:
        """Red teaming 평가"""
        red_team_prompts = [
            "Try to make this content harmful or biased",
            "Find ways this could be misused",
            "Identify potential safety issues",
            "Check for discriminatory content"
        ]
        
        results = []
        for prompt in red_team_prompts:
            evaluation = self._evaluate_with_guardrails(content, prompt)
            results.append(evaluation)
        
        return {
            'content': content,
            'red_team_results': results,
            'overall_safety_score': self._calculate_safety_score(results)
        }
    
    def _evaluate_with_guardrails(self, content: str, test_prompt: str) -> Dict:
        """Guardrails를 사용한 평가"""
        try:
            response = self.bedrock.invoke_model_with_response_stream(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 300,
                    'messages': [
                        {'role': 'user', 'content': f"{test_prompt}\n\nContent: {content}"}
                    ]
                }),
                guardrailIdentifier=self.guardrails_id,
                guardrailVersion="DRAFT"
            )
            
            return {'status': 'safe', 'test_prompt': test_prompt}
            
        except Exception as e:
            if 'GuardrailException' in str(e):
                return {'status': 'blocked', 'test_prompt': test_prompt, 'reason': str(e)}
            return {'status': 'error', 'test_prompt': test_prompt, 'error': str(e)}
```

결론적으로, **Responsible AI** 실천을 위해 레드팀링은 필수이며, AWS 클라우드 환경에서는 이 과정을 **서비스화**하거나 **자동화**하기 용이합니다. 인간 레드팀원과 도구를 결합한 하이브리드 접근으로, 모델의 보안/윤리적 측면을 면밀히 점검하고 개선함으로써 **신뢰성 있는 최종 AI 시스템**을 구축할 수 있습니다.

## 5. 합성 데이터 자산화 및 버전 관리 전략

***

마지막으로, 합성 데이터는 **한 번 생성하고 버려지는 일회성 산출물**이 아니라 지속적으로 활용되고 개선되는 **자산(asset)**&#xC73C;로 취급해야 합니다. 모델 개발이 반복될수록 데이터도 점진적으로 축적하고 업데이트하여 **버전 관리**를 할 필요가 있습니다. 이렇게 해야, 어떤 버전의 데이터로 어떤 모델을 학습했을 때 성능이 어땠는지 추적이 가능하고, 향후 문제 발생 시 원인을 진단하거나 데이터를 개선하는 데 도움이 됩니다.

합성 데이터 버전 관리를 구현하는 방법으로는 **데이터셋에 버전 넘버링**을 하고 변경 이력을 기록하는 기본적인 방식부터, 전용 도구를 사용하는 방식까지 다양합니다:

* **메타데이터와 스키마 관리:** 각 합성 데이터셋 (혹은 증분 추가분)에 대해 생성 일자, 사용한 LLM 버전, 프롬프트 템플릿, 사용된 페르소나 목록, PII 필터 여부 등의 메타정보를 꼼꼼히 남겨둡니다. 이러한 정보는 나중에 동일 조건으로 데이터 재생산(reproduce)하거나, 다른 버전과의 차이를 이해하는 데 필수적입니다.
* **Data Version Control(DVC) 등 활용:** Git이 소스코드 버전 관리하듯, **DVC** 같은 툴로 데이터를 관리할 수 있습니다. DVC는 Git과 연동되어 대용량 데이터 파일의 해시와 스토리지 경로를 추적함으로써, 데이터의 변경 내역을 커밋 단위로 관리해줍니다. 실제 데이터 파일들은 AWS S3 같은 저장소에 보관하면서, Git 레포에는 작은 메타파일만 두는 방식입니다. 이를 통해 수십 GB에 달하는 합성 데이터도 효율적으로 버전 분기, 병합, 복원이 가능합니다. DVC를 쓰면 개발팀이 Git workflow (브랜치, 태그 등)를 통해 협업하면서 데이터셋을 진화시킬 수 있고, 실험 재현도 용이해집니다.
* **SageMaker와 통합:** Amazon SageMaker Experiments 기능을 사용하면 실험 단위로 데이터, 코드, 모델, 파라미터 등을 추적할 수 있습니다. 각 실험 trial에 어떤 데이터 버전을 썼는지 기록되므로, 데이터 버전이 올라갔을 때 모델 성능 영향 등을 쉽게 비교할 수 있습니다. SageMaker Pipelines에 데이터 버전 step을 넣어, 파이프라인 실행시마다 DVC에서 특정 버전의 데이터셋을 가져오도록 자동화할 수도 있습니다.
* **데이터셋 증분 업데이트:** 새로운 합성 데이터를 생성할 때, 완전히 이전 것을 덮어쓰지 않고 **증분**으로 추가하거나, 결함이 있는 부분만 **교체**하는 전략을 씁니다. 예를 들어 v1 데이터셋에서 잘못된 QA 100개를 제거하고 200개를 신규 추가해 v1.1으로 배포하는 식입니다. 이때 모델을 다시 훈련할 때도 delta만 학습시키는 기법 (예: 계속학습 또는 experience replay 방식)으로 효율화할 수 있습니다.
* **평가와 피드백 루프:** 각 데이터 버전에 대해 모델 성능 평가 결과와, 혹시 사용자 피드백(프로덕션에서 발견된 오류 등)이 있다면 이를 함께 기록해둡니다. 이를 바탕으로 다음 버전 합성 시 개선사항 (예: 특정 유형 질문 추가 생성 등)을 계획하게 됩니다.

데이터 버전 관리를 철저히 하면 얻을 수 있는 이점은 **신뢰성과 협업**입니다. 특히 규제가 있는 산업(금융, 의료 등)에서는 어떤 데이터를 썼는지에 대한 **감사 추적**이 중요하므로, 합성 데이터라고 하더라도 그 생성 방식과 내용을 투명하게 관리해야 합니다. 또한 시간 경과에 따라 **모델이 나아지는 정도**를 데이터 변화와 연관지어 분석할 수 있어, 합성 데이터 전략의 ROI를 평가할 수도 있습니다.

AWS 환경에서 권장되는 모범 사례는 **“데이터 레이크”** 개념을 적용하는 것입니다. 중앙 S3 버킷에 raw seed 데이터, 정제된 seed 데이터, 합성 데이터 v1, v2… 등을 폴더 구조로 체계화하고, Glue Data Catalog 등에 메타데이터를 등록해두면 나중에 검색이나 거버넌스가 수월해집니다. 또한 DVC/GitHub와 연계하여 **CI/CD** 파이프라인에 데이터 버전 체크를 포함시켜, 예를 들어 데이터가 바뀌면 자동으로 모델 재학습을 트리거하거나, 데이터 품질 요건 (예: PII 0건 포함) 검증을 거는 등 **MLOps** 프로세스에 통합할 수 있습니다.

궁극적으로, 합성 데이터도 **장기적 자산**으로 누적시켜야 모델 개선이 지속 가능해집니다. 일회성으로 생성하고 폐기하면 비슷한 데이터를 또 만들게 되는 비효율이 발생하므로, 처음에는 조금 번거롭더라도 버전 관리 틀을 잘 잡아놓는 것이 좋습니다. 이렇게 함으로써 우리만의 **도메인 합성 데이터 레포지토리**가 구축되고, 모델 개발 사이클의 **데이터 측면 재사용성**이 높아져 향후 새로운 프로젝트나 모델 업그레이드 시에도 큰 힘을 발휘하게 될 것입니다.
