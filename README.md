# Korean Essay Auto-Scoring System with KoBERT and GRU

한국어 에세이 자동 채점 시스템으로, KoBERT 임베딩과 다양한 GRU 아키텍처를 활용하여 에세이를 11개 루브릭 기준으로 평가합니다.

## 🔥 주요 특징

- **KoBERT 기반 임베딩**: 한국어 에세이의 의미적 표현 학습
- **다층 GRU 아키텍처**: 4가지 모델 변형으로 성능 비교
- **UKT-A 피처 통합**: 294개의 언어학적 특징을 활용한 하이브리드 모델
- **어텐션 메커니즘**: UKT-A 피처에 대한 가중치 학습
- **11개 루브릭 평가**: 문법, 어휘, 구성, 내용 등 포괄적 채점

## 📊 실험 결과

| 모델 | Prompt 사용 | 평균 QWK | 상세 성능 |
|------|-------------|----------|-----------|
| Baseline | ❌ | **0.5284** | 기본 GRU 모델 |
| Baseline | ✅ | **0.5499** | 프롬프트 라벨링 추가 |
| GRU + LayerNorm | ✅ | **0.5984** | 정규화 및 평균 풀링 |
| GRU + LayerNorm + UKT-A | ✅ | **0.6288** | 🏆 **최고 성능** |
| GRU + LayerNorm + UKT-A + Attention | ✅ | **0.6242** | 어텐션 메커니즘 |

### 루브릭별 성능 분석 (최고 성능 모델)

| 루브릭 | QWK 점수 | 설명 |
|--------|----------|------|
| exp-grammar | 0.3083 | 문법 정확성 |
| exp-vocab | 0.3781 | 어휘 사용 |
| exp-sentence | **0.9203** | 문장 구성 |
| org-InterParagraph | 0.4801 | 단락 간 구성 |
| org-InParagraph | **0.9339** | 단락 내 구성 |
| org-consistency | **0.8899** | 일관성 |
| org-length | **0.8163** | 적절한 길이 |
| cont-clarity | 0.5071 | 명확성 |
| cont-novelty | 0.4330 | 참신성 |
| cont-description | 0.6207 | 서술력 |

## 🏗️ 시스템 아키텍처

### 1. 데이터 전처리
```
원본 에세이 → 문장 분리 → (선택적) 주제 라벨 추가 → KoBERT 토크나이징
```

### 2. 모델 구조
```
KoBERT Embedding (768dim) → Bidirectional GRU → Feature Fusion → Scoring
                                    ↓
                            UKT-A Features (294dim) → Linear → Concat
```

### 3. 4가지 모델 변형

#### Baseline
- 기본 단방향 GRU
- 마지막 히든 스테이트 사용
- UKT-A 피처 미사용

#### GRU with LayerNorm
- 2층 양방향 GRU
- LayerNorm 정규화
- 평균 풀링 적용

#### GRU with LayerNorm + UKT-A
- UKT-A 언어학적 피처 통합
- 피처 차원 축소 후 결합
- **최고 성능 달성**

#### GRU with LayerNorm + UKT-A + Attention
- UKT-A 피처에 어텐션 메커니즘 적용
- 동적 피처 가중치 학습

## 📁 파일 구조

```
├── config.py                      # 설정 및 피처 목록
├── embedding.py                   # KoBERT 임베딩 생성
├── kobert_gru_with_features.py    # 메인 모델 학습 코드
├── performance.py                 # 성능 평가 스크립트
├── dataset/
│   └── dataset_with_features.csv  # 원본 데이터셋
├── emb/                           # 임베딩 파일 저장소
├── results/                       # 결과 저장소
└── [모델명]/                      # 각 모델별 결과
    ├── topic_model.pth           # 학습된 모델
    ├── topic_y_pred_.npy         # 예측 결과
    ├── topic_y_true_.npy         # 실제 라벨
    └── attention.npy             # 어텐션 가중치 (해당시)
```

## 🚀 사용법

### 1. 환경 설정
```bash
pip install torch kobert-transformers pandas scikit-learn tqdm dask
```

### 2. 임베딩 생성
```bash
python embedding.py
```

### 3. 모델 학습
```python
# config.py에서 모드 변경
config["mode"] = "gru_with_ln_ukta"  # 원하는 모델 선택
config["is_topic_label"] = True      # 프롬프트 라벨 사용 여부

python kobert_gru_with_features.py
```

### 4. 성능 평가
```bash
python performance.py
```

## 🎯 핵심 기술

### KoBERT 임베딩
- 한국어 특화 BERT 모델 사용
- 문장별 768차원 벡터 생성
- 프롬프트 라벨링으로 맥락 정보 강화

### UKT-A 피처 (294개)
- **어휘 다양성**: TTR, RTTR, CTTR, MSTTR 등
- **품사별 분포**: 명사, 동사, 형용사 등 세부 분석
- **텍스트 응집성**: 인접 문장 간 어휘 중복도
- **구조적 특징**: 문단/문장/단어/형태소 개수

### 어텐션 메커니즘
```python
# UKT-A 피처에 대한 동적 가중치 계산
attention_scores = self.attention_weights(gru_output)
attention_weights = F.softmax(attention_scores, dim=-1)
weighted_features = ukta_features * attention_weights
```

## 📈 성능 향상 포인트

1. **프롬프트 라벨링**: 기본 모델 대비 +0.0215 QWK 향상
2. **LayerNorm + 평균 풀링**: 안정적인 학습과 일반화 성능 개선
3. **UKT-A 피처 통합**: 언어학적 특징으로 +0.0304 QWK 향상
4. **구조적 루브릭 우수성**: 문장/단락 구성 평가에서 0.9+ QWK 달성

## ⚙️ 하이퍼파라미터

```python
dropout = 0.305
learning_rate = 9.15e-4
epochs = 100
hidden_dim = 128
batch_size = 128
patience = 10  # Early stopping
max_length = 400  # 토큰 최대 길이
```

## 🔍 평가 메트릭

**Quadratic Weighted Kappa (QWK)** 사용
- 순서형 분류에 적합한 메트릭
- 예측 오차의 크기에 따라 가중 페널티 적용
- Cohen's Kappa의 확장으로 더 엄격한 평가

## 📝 주요 발견사항

1. **UKT-A 피처의 효과성**: 언어학적 특징이 딥러닝 모델 성능을 크게 향상
2. **어텐션의 한계**: 복잡한 어텐션보다 단순한 피처 결합이 더 효과적
3. **구조적 평가 우수성**: 문장/단락 구성 평가에서 뛰어난 성능
4. **문법/어휘 평가 한계**: 의미적 정확성 평가의 어려움 확인

## 🛠️ 향후 개선 방안

- **Transformer 아키텍처** 적용 검토
- **다중 태스크 학습**으로 루브릭 간 상관관계 활용
- **준지도 학습**을 통한 데이터 부족 문제 해결
- **설명 가능한 AI** 기법으로 채점 근거 제시

## 📄 License

이 프로젝트는 연구 목적으로 개발되었습니다.

---

*개발자: [Your Name] | 최종 업데이트: 2025년 8월*