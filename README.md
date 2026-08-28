# AI - 인공지능 및 신경망 사전 학습

방학 동안 학교 **인공지능** 과목 수강을 위한 사전 공부 기록입니다.
약 8주에 걸쳐 퍼셉트론부터 시작해 MLP → RNN → LSTM → 문자 단위 언어모델까지,
외부 딥러닝 프레임워크 없이 **NumPy와 C만으로 직접 구현**하며 개념을 익혔습니다.

- 기간: 2026-07 ~ 2026-08 (약 8주)
- 목표: "라이브러리를 쓰기 전에 순전파·역전파를 손으로 계산하고 코드로 재현할 수 있게 되기"
- 도구: Python(NumPy, matplotlib), C (동적 할당으로 행렬 직접 구현)

---

## 주차별 정리

| 주차 | 파일 | 주제 | 핵심 내용 |
|---|---|---|---|
| Week 01 | [`week01.ipynb`](week01.ipynb) | 퍼셉트론 · 선형 회귀 | AND/NAND/OR 퍼셉트론, XOR는 단층으로 불가 → 조합으로 구현, 최소제곱법 단순선형회귀 |
| Week 02 | [`week02.ipynb`](week02.ipynb) | 다층 신경망 손계산 | 3층 신경망 순전파(시그모이드), 오차역전파(델타 규칙)를 NumPy로 한 스텝씩 계산 |
| Week 03 | [`MLP.c`](MLP.c) | C 언어로 MLP 구현 | 이중/삼중 포인터 동적 할당으로 층·가중치·델타 배열 직접 관리, 손계산과 대조하기 위한 반올림 토글 |
| Week 04 | [`week04.ipynb`](week04.ipynb) | 파이썬으로 MLP 클래스화 | `class MLP` (순전파 / 오차 / 역전파 / train), 표준입력으로 신경망 구조를 받아 학습 |
| Week 05 | [`week05.ipynb`](week05.ipynb), [`RNN_forward.c`](RNN_forward.c) | RNN 순전파 | 은닉 상태 `h_t = tanh(Wx·x + Wh·h_{t-1} + b)` 구현, 시점별 출력 계산 |
| Week 06 | [`week06.ipynb`](week06.ipynb), [`RNN_train.c`](RNN_train.c) | RNN 역전파(BPTT) | 시점별 `cache` 저장, `grads_sum`으로 기울기 누적 후 갱신, **기울기 소실** 현상 예제 |
| Week 07 | [`week07.ipynb`](week07.ipynb) | LSTM | forget/input/output 게이트와 셀 상태 구현, RNN vs LSTM 비교 실험(장기·단기 의존성 태스크, 시드 3회 반복) |
| Week 08 | [`week08.ipynb`](week08.ipynb) | RNN 문자 단위 언어모델 | Week 06 RNN 재사용 + softmax 출력 + cross-entropy, 자기회귀 방식으로 짧은 문자열 생성 |

> Week 03 노트북 파일은 포함되어 있지 않으며, 같은 시기에 작성한 C 언어 MLP 구현(`MLP.c`)으로 대신합니다.

---

## 주차별 상세

### Week 01 — 퍼셉트론과 선형 회귀
- 가중치와 편향으로 AND·NAND·OR 게이트를 구현하고, 결정 경계를 산점도로 시각화.
- XOR는 단일 퍼셉트론(선형 분리)으로 불가능함을 확인하고, NAND·OR·AND를 조합해 해결 → **다층 구조의 필요성**을 체감.
- `MySimpleLinearRegression`: 최소제곱법으로 기울기·절편을 직접 계산해 `y = 3x + 7 + noise` 데이터를 복원.

### Week 02 — 다층 신경망 순전파 / 역전파 손계산
- 입력 3 → 은닉1 → 은닉2 → 출력 2 구조를 NumPy 행렬곱으로 순전파.
- 출력층은 항등함수, 은닉층은 시그모이드. 오차는 제곱합(SSE).
- 역전파: 출력 델타 `2(output - y)`에서 시작해 `Wᵀ·delta · sigmoid'` 로 이전 층 델타를 구하고 가중치·편향 기울기 계산.
- 이후 파이썬/C 구현 결과를 검증할 **정답지** 역할.

### Week 03 — C 언어로 MLP 구현 (`MLP.c`)
- 파이썬의 리스트/`reshape`로 쉽게 되던 부분을, C에서는 `double***` 동적 할당으로 직접 관리.
- 레이어 수·노드 수·가중치 행렬·편향·목표값·학습률·반복 횟수를 표준입력으로 받아 임의 구조의 신경망 학습.
- `transpose`, `multiply_matrix` 등 행렬 연산을 직접 작성.
- `ROUND_FOR_COMPARISON` 매크로로 소수점 2자리 반올림을 켜고 끌 수 있음 → 켜면 Week 02 손계산과 일치, 끄면 더 정밀하게 수렴(반올림 시 작은 델타가 0으로 잘려 일부 기울기가 소실되는 현상도 관찰).

### Week 04 — 파이썬으로 MLP 클래스화 (`week04.ipynb`)
- `class MLP`: `setting_input_layer`, `setting_weights`, `forward`, `error`, `backward`, `train`.
- `io.StringIO`로 표준입력을 채워 `sys.stdin.read().split()`로 파라미터를 순서대로 소비.
- C 구현과 동일한 입력 포맷을 사용해 두 구현의 결과를 맞춰 봄.

### Week 05 — RNN 순전파 (`week05.ipynb`, `RNN_forward.c`)
- `class RNN`: `forward(x, h_prev)` 에서 `h_next = tanh(Wh·h_prev + Wx·x + b)`, `cache = (x, h_prev, h_next)`.
- 길이 3짜리 시퀀스를 넣어 시점별 `h_t`, `y_t = hᵀ·Why + b_y` 출력.
- 동일 로직을 C로도 구현.

### Week 06 — RNN 역전파 / 기울기 소실 (`week06.ipynb`, `RNN_train.c`)
- 저번 주 대비 개선점:
  1. 출력값 계산을 `forward` 안으로 통합.
  2. 기울기를 시점마다 누적하는 `grads_sum` 속성 추가.
  3. `cache`가 마지막 시점만 남는 문제 → `caches` 리스트로 시점별 저장.
- **기울기 소실 예제**: 마지막 시점을 제외한 모든 `y`를 0으로 두고 역전파하면, `dh_prev = Whᵀ·dt` 가 매 시점 `tanh'` 와 `Wh` 를 반복해서 곱하면서 0으로 수렴하는 것을 확인.

### Week 07 — LSTM (`week07.ipynb`)
- 셀 상태 `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`, 은닉 상태 `h_t = o_t ⊙ tanh(c_t)`.
- 게이트(f, i, o)는 시그모이드, 후보값 `g`는 tanh. 셀 상태는 다음 시점으로 넘어갈 때 tanh를 거치지 않아 기울기가 보존됨.
- **RNN vs LSTM 비교 실험** (교수님 피드백 반영):
  - 장기 의존성(copy task) / 단기 의존성 태스크, 시퀀스 길이 T = 5·15·30, 시드 3회 반복 후 평균±표준편차.
  - 결과: 이 작은 규모에서는 평균 손실 자체는 RNN이 더 낮았지만(게이트가 시그모이드 초기값 ~0.5에서 정보를 절반쯤 걸러내며 출발), **안정성**은 LSTM이 우위 — T=30·학습률 0.1에서 RNN은 발산, LSTM은 표준편차가 훨씬 작음.
  - 결론: LSTM의 장점은 "항상 더 좋은 결과"가 아니라 **기울기 소실을 완화해 안정적인 학습**을 제공하는 것.

### Week 08 — RNN 문자 단위 언어모델 (`week08.ipynb`)
- Week 06의 RNN 은닉 상태 계산을 그대로 두고 출력층과 손실만 교체:
  - 출력: `Whyᵀ·h` 에 softmax → 다음 글자 확률분포.
  - 손실: cross-entropy `-log(정답 글자 확률)`.
  - 역전파: `dz = 예측확률 - 정답원핫` 이후는 회귀 버전과 동일.
- 원-핫 인코딩, many-to-many(입력 `"hell"` → 정답 `"ello"`), 자기회귀 생성(샘플링/argmax).
- `"hello world"` 학습: loss 2.12 → 0.009, argmax 생성 시 문자열 정확히 복원. 확률적 샘플링은 낮은 확률 글자가 뽑혀 가끔 흐트러짐.
- 은닉 차원 `H`를 4로 줄이면 기억 용량이 부족해 성능 저하, 단 `"hello hello ..."` 처럼 반복적인 패턴은 작은 `H`로도 잘 학습됨.

---

## 파일 구조

```
AI/
├── week01.ipynb      퍼셉트론, 단순선형회귀
├── week02.ipynb      다층 신경망 순전파/역전파 손계산
├── week04.ipynb      파이썬 MLP 클래스
├── week05.ipynb      RNN 순전파
├── week06.ipynb      RNN 역전파(BPTT), 기울기 소실
├── week07.ipynb      LSTM, RNN vs LSTM 비교 실험
├── week08.ipynb      RNN 문자 단위 언어모델
├── MLP.c             C 언어 MLP 구현 (동적 할당)
├── RNN_forward.c     C 언어 RNN 순전파
└── RNN_train.c       C 언어 RNN 순전파 + BPTT + 가중치 갱신
```

## 실행 방법

### 노트북 (Python)
```bash
pip install numpy matplotlib
jupyter notebook
```

### C 파일
```bash
gcc MLP.c -o mlp -lm
./mlp < input.txt    # 입력 포맷은 각 파일 상단 주석 참고

gcc RNN_forward.c -o rnn_forward -lm && ./rnn_forward
gcc RNN_train.c -o rnn_train -lm && ./rnn_train
```

`MLP.c` 표준입력 포맷: `레이어 수 → 각 층 노드 수 → 입력값 → 층 사이 가중치 행렬 → 편향 → 목표값 → 학습률 → 반복 횟수`

---

## 8주간 배운 것

- 퍼셉트론의 한계(XOR) → 다층 구조 → 오차역전파로 이어지는 흐름을 직접 구현하며 이해.
- 순전파·역전파를 손으로 계산한 뒤 Python·C 두 언어로 재현해 결과를 대조.
- RNN에서 시퀀스가 길어질 때 기울기가 소실되는 이유를 코드로 확인하고, LSTM의 셀 상태가 이를 어떻게 완화하는지 실험으로 검증.
- 회귀(제곱오차)에서 분류(softmax + cross-entropy)로 출력층만 바꿔 언어모델을 만들 수 있음을 확인.
