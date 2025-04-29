### 허깅페이스에서는 모델의 바디 부분과 헤드 부분 별도로 가져올 수 있는거임? 각각의 역할은?

바디 부분의 출력은 입력 -> 트랜스포머 모델 내부를 거쳐서 나온 고차원된 벡터 표현으로 문장에 대한 의미가 압축된 벡터 표현임. (보통 CLS 토큰으로 시작)

아니면 문장 내의 각 토큰들의 상호 작용, 문맥을 반영한 임베딩이거나 

헤드는 이런 바디의 출력을 피처로 사용해서 작업을 하는 걸 말함. 분류 같은 것. 


### 바디가 반환하는 여러 잠재 상태는 뭐가 있음? 

Transformer 바디(예: BertModel, RobertaModel 등)를 output_hidden_states=True·output_attentions=True 로 설정하면 잠재 상태들을 볼 수 있음. 

last_hidden_state: 
- 마지막 레이어의 토큰 임베딩 값으로, downstream 헤드 부분에서 이 값을 주로 사용함. 

pooler_output: 
- BERT 계열에서는 \[CLS] 위치의 벡터를 선형+tanh 층에 통과시켜 만든, 문장 전체 대표 벡터

past_key_values: 
- 디코더 모델(GPT-2, T5 디코더 등)을 사용할 때, 이미 계산된 각 레이어의 key/value 캐시
- 이전 시점의 어텐션 계산 결과를 빠르게 사용하고 싶을 때 사용 


### HuggingFace 에서는 Traniner API 를 이용해서 모델을 학습시킬 수 있는거임? 이 방법 말고 학습 시키는 방법은? 

이 방법 말고는 직접 PyTorch 훈련 루프 작성 하는 것이 있음. 

torch.utils.data.DataLoader → model(input) → loss.backward() → optimizer.step() 순으로, 완전 커스텀하게 학습 과정을 제어하는 거임. 

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
dataloader = DataLoader(dataset["train"], batch_size=16, shuffle=True)

model.train()
for epoch in range(3):
    for batch in dataloader:
        inputs = tokenizer(batch["sentence1"], batch["sentence2"],
                           return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs, labels=batch["label"])
        loss = outputs.loss
        loss.backward()
        # 계산된 그래디언트로 모델 파라미터 업데이트
        optimizer.step()
        # 다음 배치를 위해 그래디언트 초기화
        optimizer.zero_grad()
```


### Trainer API 를 이용해서 학습을 할 때 신경써야 하는 요소들에 대해서 정리해줘

1) 데이터 전처리(Data & Preprocessing)
- 토크나이저 설정
    - pad_to_multiple_of, truncation, padding 방식(“longest” vs “max_length”)
        - truncation: 너무 긴 시퀀스를 잘라 길이를 제한
        - padding: 짧은 시퀀스를 패딩 토큰으로 채워 길이를 맞춤 
        - pad_to_multiple_of: 패딩 후 길이를 특정 배수로 맞춰서 메모리 연산 최적화 
        - DataCollatorWithPadding: 배치 별로 이 옵션을 잘해주는 거임. 
    - 배치 내 패딩 최적화용 DataCollatorWithPadding 활용

- 데이터셋 분할
    - train/validation/ test 비율 적절히(예: 80/10/10)
    - 클래스 불균형 시 stratify 적용 (데이터 셋을 분할했을 때 소수의 클래스는 포함되지 않는 경우를 막기 위함임.)

- 증강·샘플링
    - 데이터가 부족하거나 불균형한 경우 증강(AEDA, back-translation) 혹은 오버/언더샘플링


2. 모델·토크나이저 초기화
- 사전학습 모델 선택
    - 태스크 특성에 맞는 구조(encoder-only vs encoder-decoder vs decoder-only)
- 헤드 수정
    - num_labels, problem_type 등 올바르게 지정
    - num_labels: 헤드가 분류해야하는 클래스 개수 
    - problem_type: 헤드가 수행해야하는 문제 유형 

- 토크나이저-vocab 일치
    - 토크나이저 보크와 모델 임베딩 크기(resize_token_embeddings) 일치 여부 확인