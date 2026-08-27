# 숫자 감사 (paper.tex v2 기준, 2026-08-26)

## 🚩 OUTDATED — 수정 필요
1. **"40 consecutive frames ... first 64 frames" (Tasks)** — v02b 프로토콜. 현재 사다리 런·콜랩 v0.3 전부 **64프레임 윈도우** (--frames 64 --first_frames 64, 3.2s). "40-frame length matches ARDY's two-second window" 근거도 함께 소멸. → 재작성.
2. **"0.85-GB $64^2$ tier" (abstract, intro, related work ×2)** — README/HF 카드/자체 테이블 모두 **0.79 GB**. → 전부 0.79로.
3. **Accessibility "smaller training tensor (40-frame windows)"** — 콜랩 v0.3은 64프레임. → 수정.

## ⚠️ 경미한 불일치 — 플래그만
4. real–real FVD: 테이블 115.2 (fvd_64f_n128.json의 a-vs-b), run_ckpt 내부 레퍼런스는 114.7. 같은 매니페스트, 코드 경로 차이 ~0.5. 모델 행들은 run_ckpt 산출이므로 엄밀히는 114.7이 짝. → 일단 115.2 유지 + 각주 없음; Jin 판단 필요하면 114.7로 통일.
5. reverse_time FVD: 120f 스터디 79.3 vs 64f 파일 120.5 — 프로토콜이 달라서 불일치 아님 (페이퍼는 120f만 인용). OK.

## ✓ 검증 통과 (소스 대조)
- 데이터: 1,340 모션 / 4,020 비디오 / 482,400 프레임 / 134 프롬프트 (143 생성, 9 제외) / 스플릿 1,072·134·134 / 3,216·402·402 / 385,920·48,240·48,240 / mini 0.79GB·frames 4.4GB·motion 0.33GB / 27조인트 / 20fps / 6s / 120프레임.
- 120f 스터디: 80.5 / 465.5 / 419.2 / 328.5 / 79.3; 속도 .313/.299; 가속 .348→2.858; 저크 .060→.384; 루프 .507/.084; 리버설 −1.65±2.69 (30회).
- 구조 corruption: LIE +.203, TVR +.085, 500 rerenders.
- 사다리 (오늘 산출): single-frame real .115/.093/.039, image DiT .128/.050/.030; win64 real .116/.093/.037/.373/.501/.073; codec floor .137/.104/.037/.380/.503/.104/124.4; rand .942/.035/.051/.408/.559/.326/1641.3; warm .169/.045/.034/.850/.719/.324/545.0; seed repeat .183/570.7; latent 503.3/527.2; train_replay 125.7.
- 리소스: 0.10s/step·10.5GB (image), 3.4s (2-run 공유), 1.07s (local3d 솔로), no-ckpt 55.1GB·0.82s. RTX PRO 6000 96GB.
- 검증: 514,800 프레임 / 1,430 모션 / 99.88·99.92·99.50·99.99%.
- ST 비용: 4.7×, 0.17 vs 0.19 s/step. 백본 39.9M.
- 대기 중: local3d·fullst 10k 재평가 (지금 도는 중) → 두 행 + ladder 문단.
