---
icon: cloud-binary
coverY: 0
---

# Amazon Bedrock AgentCore

{% hint style="danger" %}
_Disclaimer: 본 가이드는 Amazon Bedrock AgentCore를 처음 접하는 분들을 위한 100-200레벨의 개요와 getting started 및 코드 스니펫을 제공합니다. 본 가이드는 개인적인 의견과 가이드로 AWS의 공식 문서를 대표하는 것이 아니며, 모든 내용은 AWS의 공식 문서를 우선으로 합니다._

Contributors

* Main Contributor: Daekeun Kim ([daekeun@amazon.com](mailto:daekeun@amazon.com))
{% endhint %}

## Overview

AWS Bedrock AgentCore는 프로덕션 환경에서 AI 에이전트를 대규모로 배포하고 운영할 때 겪는 다양한 문제를 해결하기 위해 설계된 서비스입니다. 기존의 Bedrock Agents가 에이전트를 쉽게 만들고 실험하는 데 초점을 맞췄다면, AgentCore는 여기서 한 단계 나아가 다음과 같은 차별점을 제공합니다.

* **보안 및 안정성:** AgentCore Runtime은 세션 격리가 완벽한 보안 서버리스 실행 환경을 제공하여 데이터 유출을 방지합니다. 또한, 체크포인트 및 복구 기능을 통해 예기치 않은 중단이나 실패 시에도 에이전트가 원활하게 복구될 수 있도록 합니다.
* **확장성:** 수천 개의 동시 세션으로 자동 확장될 수 있어 인프라 용량 계획이 필요 없습니다. 또한, 8시간까지 실행 가능한 비동기 워크로드를 지원하여 장기 실행 작업에도 적합합니다.
* **모듈형 서비스:** AgentCore는 독립적으로 사용 가능하며, 상호 연동되도록 최적화된 여러 모듈형 서비스로 구성되어 있습니다.
  * **AgentCore Runtime:** 에이전트를 안전하게 배포하고 확장합니다.
  * **AgentCore Memory:** 단기 및 장기 메모리를 관리하여 에이전트가 여러 상호작용에 걸쳐 컨텍스트를 유지할 수 있도록 합니다.
  * **AgentCore Gateway:** 기존 API나 AWS Lambda 함수를 에이전트 도구로 변환하여 에이전트가 다양한 시스템과 통합될 수 있도록 합니다.
  * **AgentCore Identity:** 에이전트가 AWS 서비스 및 타사 도구에 안전하게 접근할 수 있도록 권한을 관리합니다.
  * **AgentCore Tools - Code Interpreter:** 에이전트가 격리된 샌드박스 환경에서 코드를 안전하게 실행할 수 있게 해줍니다.
  * **AgentCore Tools - Browser:** AI 에이전트가 대규모로 웹사이트와 상호 작용할 수 있도록 빠르고 안전한 클라우드 기반 브라우저 런타임을 제공합니다.
  * **AgentCore Observability:** 에이전트의 동작을 단계별로 시각화하고 디버깅, 모니터링을 지원합니다.
* **유연성:** 오픈 소스 프레임워크(예: LangGraph, CrewAI)나 커스텀 프레임워크, 그리고 모든 모델과 함께 사용할 수 있어 개발 팀의 기존 도구를 유지하면서 엔터프라이즈급 기능을 추가할 수 있습니다.

이러한 기능들을 통해 개발자들은 보안, 메모리 관리, 모니터링 등 인프라 구축에 드는 'undifferentiated heavy lifting' (차별화되지 않는 고된 작업)에서 벗어나 에이전트의 핵심 기능 개발에 집중하여 프로덕션 환경에 맞는 솔루션을 더 빠르게 시장에 출시할 수 있습니다.

## 세부 서비스 살펴보기

{% columns %}
{% column width="50%" %}
{% content-ref url="agentcore-runtime.md" %}
[agentcore-runtime.md](agentcore-runtime.md)
{% endcontent-ref %}

{% content-ref url="agentcore-identity.md" %}
[agentcore-identity.md](agentcore-identity.md)
{% endcontent-ref %}

{% content-ref url="agentcore-code-interpreter.md" %}
[agentcore-code-interpreter.md](agentcore-code-interpreter.md)
{% endcontent-ref %}
{% endcolumn %}

{% column width="50%" %}
{% content-ref url="agentcore-gateway.md" %}
[agentcore-gateway.md](agentcore-gateway.md)
{% endcontent-ref %}

{% content-ref url="agentcore-observability.md" %}
[agentcore-observability.md](agentcore-observability.md)
{% endcontent-ref %}

{% content-ref url="agentcore-browser.md" %}
[agentcore-browser.md](agentcore-browser.md)
{% endcontent-ref %}
{% endcolumn %}
{% endcolumns %}

{% content-ref url="agentcore-memory.md" %}
[agentcore-memory.md](agentcore-memory.md)
{% endcontent-ref %}

## References

* Amazon Bedrock AgentCore 공식 문서: [https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)
* Strands Agents 문서: [https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/](https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/)
* 코드 예제: [https://github.com/awslabs/amazon-bedrock-agentcore-samples/](https://github.com/awslabs/amazon-bedrock-agentcore-samples/)
