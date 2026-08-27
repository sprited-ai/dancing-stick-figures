# Paper 3차 패스 — 문단별 코멘트 (2026-08-26 저녁)

기준: (a) 이 문단이 없으면 독자가 뭘 잃는가? (b) 주장인가 장식인가? (c) 다른 문단과 중복인가?

## Abstract
- "Training a video generation model is hard to learn by doing" — 첫 문장 어색 (train을 learn by doing?). → "Learning video generation by training models yourself is hard."로 교정.
- 숫자 나열 (1,340/4,020/482,400) — 두 개면 충분. 482,400은 테이블에 있음. → 유지하되 한 번만.
- FVD 79.3/80.5 리버설 결과 — 페이퍼의 가장 정직한 발견이라 abstract에 유지. **KEEP**
- 마지막 문장 릴리즈 목록 — "checkpoints, and a Colab lesson" 정도로 압축. **TIGHTEN**

## Introduction
- 3-불릿 (Data/Experimentation/Evaluation) — 동기의 뼈대. 각 불릿 마지막 문장들이 사족 (예: "Retraining a controlled variant in an afternoon is not." — 수사적 반전, 슬롭 냄새). → 불릿당 1–2문장으로. **TIGHTEN**
- RQ 문단 — 필요. "That requires three things..." 이후 세 문장은 디자인 크라이테리아 재진술 — 불릿과 중복 아님 (불릿=문제, 여기=요구조건). **KEEP, 압축**
- "Dancing Stick Figures is built to those constraints..." 문단 — dataset/suite/models 3역할 소개. contributions 리스트와 절반 중복. → 이 문단을 2문장으로 줄이고 리스트가 세부를 맡게. **MERGE-DOWN**
- Contributions 리스트 — 표준 관행, 링크 역할. **KEEP**

## Related Work
- 도입 2문장 — "combines data, renderer, models, measurements in one experiment" 는 이미 인트로에서 말함. → 1문장으로. **TIGHTEN**
- Small/controlled video — 위치 정립에 필요. **KEEP**
- Synthetic people — 좋음, 이미 짧음. **KEEP**
- Training data for video generation — 뒷문장 (zhao/jin2026) "different compute regime" 논지 유효. **KEEP**
- Human motion as source data — 가장 약한 문단. AMASS/HumanML3D는 우리와 경쟁하지 않고, 요지는 "모션은 소스 재료"뿐. → 2문장으로 축소 (삭제는 인용 유지 위해 안 함). **SHRINK**
- Evaluation of generated video — 필요, 이미 압축됨. **KEEP**

## Dataset
- 생성+큐레이션 문단 — 9개 제외 프롬프트 예시 열거가 길다. 예시는 3개면 전달됨; 전체 목록은 릴리즈에 있다고 명시돼 있음. → 예시 축소. **TIGHTEN**
- 파라미터/버퍼 문단 — 데이터셋 카드의 핵심. **KEEP**
- 품질 플래그 문단(frozen/levitation) — 2문장, 릴리즈 정직성. **KEEP** (tiers 문단과 합침)
- 컴퓨트 티어 문단 — 3문장 → 2문장. **TIGHTEN**
- 리빌드 문단 — Accessibility 섹션의 리빌드 문단과 **중복**. → 여기선 1문장 예고만 남기고 상세는 Accessibility로. **DEDUP**

## Tasks and failure diagnostics
- Video task (윈도우 뷰) — 로드베어링 (40프레임/first-64 정당화). ARDY 페이싱 설명 중복 조심 (Limitations에도 나옴) → 여기 1회로 유지, Limitations 쪽을 줄임. **KEEP**
- Structural answer key 리스트 — 페이퍼의 심장. **KEEP**
- 비디오 신호 문단 — two-sided 설명 필요. **KEEP**
- real reference/FVD 문단 — 필요. **KEEP**
- Controlled validation — 필요. **KEEP**

## Results
- Metric stress test — 수치 밀도 높지만 전부 주장 근거. **KEEP**
- Released reference models 1문단 (백본) — **KEEP**
- 사다리 rung 설명 문단 — 길다. "Image-first is established practice... no novelty claim" 정직성 유지. 4.7× 비용 문장 유지. **TIGHTEN 소폭**
- 코덱 감사 문단 — 페이퍼에서 제일 자랑스러운 정직성. 다만 "planned for a revision" 문장은 Limitations로 이동해도 됨 → 유지(문맥상 여기가 자연스러움). **KEEP**
- 테이블 각주 — train_replay 125.7 설명이 두 번 나올 뻔한 것 정리돼 있음. **KEEP**
- Reading the ladder — mixing rung 결과 나오면 2문장 추가 예정. **KEEP+EXTEND**
- Why pixel space — 코덱 감사와 논지 연결됨. 3문장 유지. **KEEP**

## Accessibility
- 릴리즈/노트북 문단 — **KEEP**
- 리빌드 검증 문단 — 퍼센트 6개 나열. 99.88/99.92/99.50/99.99 중 대표 2개+"나머지는 검증 리포트에" 로 줄여도 되지만, 재현성 섹션의 존재 이유가 이 숫자들임. → 유지하되 문장 수만 축소. **TIGHTEN**
- 리소스 문단 — RQ와 직결. **KEEP**
- 라이선스 문단 — 필수. **KEEP**

## Limitations
- 1문단 — ARDY 페이싱 문장이 Tasks와 중복 → 절반으로. **TIGHTEN**
- rig estimator 문단 — "다음 확장" 하나만 말하면 됨. 4문장 → 2문장. **TIGHTEN**

## Conclusion
- 마지막 문장 "It is a small, complete first experiment..." — abstract 마지막과 동일 문구였던 것 (v2에서 abstract쪽은 제거됨). **KEEP**
- AI 사용 문단 — **KEEP**

## 종합
- 삭제급: 없음 (v2에서 이미 제거됨). 이번 패스는 압축·중복 제거 중심, 예상 -0.5~1p.
- 구조 변경: Dataset 리빌드 상세 → Accessibility로 단일화.
