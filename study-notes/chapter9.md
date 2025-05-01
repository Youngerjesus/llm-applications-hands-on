### LLM 캐시란 무엇인지? 

LLM 추론 비용을 줄이기 위해 이전에 받은 요청이나 이와 비슷한 요청이 오면 캐시에 저장된 응답을 전달하는 기법임. 

LLM 캐시는 크게 두 가지 방식으로 동작할 수 있음. exact match 방식이거나 similar search 방식 

유사도 방식에서는 요청의 유사성을 비교하기 위해서 임베딩을 이용함.


### LLM 이 답변하지 않아야 하는 요청을 어떻게 구별하고 판단할 것인지? 

크게는 다음과 같은 방법들이 있음: 
- 규칙 기반(Rule-based) 
- 임베딩 이용 (답변하지 말아야 할 내용이 있다면 이를 임베딩으로 만들어두고 검색하는 방법임)
- LLM 활용 (답변 전에 LLM 에게 물어보는 거임)
- 분류 또는 회귀 모델을 학습해서 사용 

실제 구현에서는 NeMo-Guardrails 라이브러리를 활용하는 방법이 있음

NeMo Guardrails란 무엇인가?
- 오픈소스 툴킷으로, LLM 기반 대화 애플리케이션에 “프로그래머블 가드레일(guardrails)”을 삽입해
- 원치 않는 주제를 차단하거나
- 사전 정의된 대화 흐름을 강제하거나
- 출력 형식을 구조화하거나
- 외부 서비스(툴)와의 안전한 연동을 가능하게 합니다.

아키텍처 구성요소
- Rails (정책 단위)
	- YAML 또는 Colang(.co)으로 정의
	- 종류: Input, Dialog, Retrieval, Execution, Output
- Fences (검문소)
    - 텍스트 전/후처리 시 정규표현식 또는 LLM 분류기로 민감 콘텐츠 탐지
- Handlers
    - allow / warn / block / redirect 등의 액션 구현
- Safety Spec
    - 민감도 임계치, 블랙·화이트리스트 등

NeMo Guardrails에서 지원하는 다섯 가지 주요 Guardrails 유형: 
- Input Rails: 사용자가 보낸 메시지가 LLM으로 전달되기 전
- Dialog Rails: LLM을 어떻게 호출할지 결정하는 단계 (대화 흐름에 따라 발화 순서를 정의)
- Retrieval Rails: 외부 문서·지식베이스에서 RAG(검색 후 생성)용 컨텍스트를 가져온 뒤 부적절하거나 불필요한 정보 필터링
- Execution Rails: LLM이 호출하도록 지시된 외부 액션(툴·함수)을 실행하기 전후
- Output Rails: LLM 이 생성한 응답을 최종 사용자에게 던지기 전에 


### LLM 을 모니터링 하는 방법은? 

LLM 은 입력에 대한 출력 같은걸 로깅으로 남겨더야함

대표적인 로깅 도루로는 W&B, MLflow, PromptLayer 등이 있음.