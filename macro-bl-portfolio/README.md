# Macro-Driven Black-Litterman Portfolio Optimization Pipeline

거시경제 팩터, DNS 금리기간구조 모형, 시나리오 기반 기대수익률, Black-Litterman 최적화를 결합한 포트폴리오 최적화 파이프라인입니다.

## 1. 주요 기능

- ETF 가격 데이터 수집 및 월별 수익률 계산
- FRED 기반 매크로 팩터 구성
  - 기대인플레이션
  - 실질금리
  - 신용스프레드
  - 성장
  - 노동시장
  - 원자재
- 자산별 팩터 회귀 및 회귀 진단
- DNS + Kalman Filter 기반 금리기간구조 추정
- Risk-On / Risk-Off 시나리오별 기대수익률 산출
- R² 기반 기대수익률 및 공분산 블렌딩
- Black-Litterman 기반 포트폴리오 최적화
- 시각화 결과 PNG 저장

## 2. 프로젝트 구조

```text
macro-bl-portfolio/
├─ src/
│  └─ macro_bl_pipeline.py
├─ outputs/
│  └─ .gitkeep
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ config.example.env
```

## 3. 설치 방법

```bash
pip install -r requirements.txt
```

선택적으로 FRED API 키를 사용할 수 있습니다. `config.example.env`를 참고해 로컬에서 `.env` 파일을 만들거나, 운영체제 환경변수로 설정하면 됩니다.

```bash
FRED_API_KEY=your_fred_api_key_here
OUTPUT_DIR=outputs
```

Windows PowerShell 예시는 다음과 같습니다.

```powershell
$env:FRED_API_KEY="your_fred_api_key_here"
$env:OUTPUT_DIR="outputs"
```

## 4. 실행 방법

```bash
python src/macro_bl_pipeline.py
```

실행 후 결과 이미지는 기본적으로 `outputs/` 폴더에 저장됩니다.

## 5. 주의사항

- `.env` 파일과 API 키는 깃허브에 커밋하지 않습니다.
- `outputs/` 폴더의 산출물은 기본적으로 커밋 대상에서 제외됩니다.
- yfinance 또는 FRED 연결이 실패하는 경우 일부 데이터는 코드 내 fallback 로직에 따라 합성 데이터로 대체될 수 있습니다.
- 본 코드는 투자 권유가 아니라 학습 및 연구 목적의 정량 분석 예시입니다.
