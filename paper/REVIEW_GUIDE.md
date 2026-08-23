# paper-m6 리뷰 가이드 (Jin용, 2026-08-23)

브랜치 `paper-m6`, 커밋 20개, 페이퍼는 7쪽 (`output/pdf/paper.pdf`).
`paper.tex`는 이 브랜치에서 신설 — diff가 아니라 통독 대상.

## 30분 리뷰 동선

1. **1쪽 abstract + intro 포지셔닝** (5분) — "테스트베드" 프레이밍이 네 의도와
   맞는지. 저자 표기 Jin Hyuk Cho + ORCID 확인.
2. **핵심 신규 섹션** (15분), 이번 세션에서 들어간 것:
   - "Latent long-horizon track: a frozen codec" — 코덱 선택·고정 근거
   - "Block-autoregressive M6 and its convergence trade-off" + fig:m6 —
     동결 곡선 (100k 점선 포함)
   - **"Breaking the freeze"** + fig:m6fix — 파일럿 5개 표 + v8 결과.
     이 문단이 논문의 새 하이라이트.
   - "Full-clip latent control R0" — 50k 곡선 확장 문장까지
3. **status 테이블 + future work** (5분) — 주장 한계 서술이 과하지도
   부족하지도 않은지.
4. **결정** (5분): arXiv go/no-go. go면 남는 작업: 최종 페이지 QA 한 번,
   (선택) v8 300k 결과 한 문장 반영, 제출 메타데이터.

## 지금 페이퍼에 안 들어간 것 (의도적)

- v9 리그 공동생성 트랙 — 다음 논문/v0.2 척추로 보류
- CFG 스윕, t5-base 기각, 약점 택소노미 — EXPERIMENT_LOG에만 기록
- v8 300k — 완주 시 예측 대조 후 한 문장 추가 여부만 판단

## 증거 상태 요약

| 주장 | 근거 | 한계 |
|---|---|---|
| 동결은 구조적 | h8 100k: TVR .097(<floor) & speed .115 고정 | 1 seed |
| 원인은 gradient 배분 | 파일럿 5개: 신호재배분 3승 / 히스토리 2패 | 각 2k·1 seed |
| v8이 Pareto 탈출 | 100k: TVR .122 + speed .286 (real 91%) | 1 seed, 성분귀속은 파일럿 의존 |
| R0는 구조 정체 | 50k: TVR .22 평탄 | 예산 상이 (50k vs 100k) |

전 곡선 데이터: `paper/results/m6_h8_milestones_n64.json`
전체 이력: `paper/EXPERIMENT_LOG.md`
