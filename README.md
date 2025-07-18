# 🎮 AI 테트리스 게임

Next.js와 Python을 사용한 AI 제어 테트리스 게임입니다.

## 🚀 설치 및 실행

### 1. Next.js 게임 서버 설정

```bash
# 프로젝트 루트 디렉토리에서
pnpm install
pnpm dev
```

브라우저에서 http://localhost:3000 접속

### 2. Python AI 설정

```bash
# Python 패키지 설치
pip3 install requests numpy

# python_ai 디렉토리로 이동
cd python_ai
```

## 🎯 사용법

### API 방식 (권장) 🌟

**1단계: 게임 서버 실행**
```bash
pnpm dev
```

**2단계: 브라우저에서 게임 접속**
- http://localhost:3000 접속
- 화면에서 "🤖 AI 모드 ON" 버튼 클릭

**3단계: Python AI 실행**
```bash
# 간단한 랜덤 AI
python3 simple_tetris_api.py

# 고급 AI (휴리스틱 기반)
python3 tetris_api_client.py
```

### 🎮 게임 제어

- **수동 모드**: 키보드로 직접 조작
  - ← → : 이동
  - ↑ : 회전
  - ↓ : 빠른 드롭
  - Space : 하드 드롭
  - P : 일시정지

- **AI 모드**: Python이 자동 제어
  - 웹 UI에서 "AI 모드 ON" 버튼 클릭
  - Python 스크립트가 HTTP API로 게임 조작

## 🤖 AI 알고리즘

### 간단한 AI (`simple_tetris_api.py`)
- 랜덤 액션 선택
- 기본적인 연결 테스트용

### 고급 AI (`tetris_api_client.py`)
- 휴리스틱 기반 평가 함수
- 보드 높이, 구멍, 완성 라인 고려
- 실시간 점수 모니터링

#### 평가 요소:
- **높이**: 전체 블록 높이 (낮을수록 좋음)
- **구멍**: 블록 아래 빈 공간 (적을수록 좋음)
- **완성 라인**: 채워진 라인 (많을수록 좋음)
- **높이 차이**: 열 간 높이 차이 (작을수록 좋음)

## 🔧 API 엔드포인트

### `GET /api/game`
게임 상태 조회
```json
{
  "success": true,
  "data": {
    "board": [[...], [...]],
    "currentPiece": {...},
    "score": 1500,
    "level": 2,
    "isGameOver": false
  }
}
```

### `POST /api/game`
액션 전송
```json
{
  "type": "action",
  "action": "left" // "right", "down", "rotate", "drop", "pause"
}
```

## 📊 특징

✅ **웹 기반 시각화**: 브라우저에서 실시간 게임 관찰  
✅ **API 통신**: HTTP 기반으로 간단한 연동  
✅ **모드 전환**: 수동/AI 모드 버튼으로 쉽게 전환  
✅ **실시간 피드백**: 점수, 레벨, 액션 수 실시간 표시  
✅ **Chrome 불필요**: 브라우저 자동화 없이 순수 API 방식  

## 🔍 트러블슈팅

**Q: 서버에 연결할 수 없습니다**
- Next.js 서버가 실행 중인지 확인 (`pnpm dev`)
- http://localhost:3000에서 게임이 로드되는지 확인
- 방화벽이 3000 포트를 차단하지 않는지 확인

**Q: AI가 동작하지 않습니다**
- 브라우저에서 "AI 모드 ON" 버튼을 클릭했는지 확인
- Python에서 requests 패키지가 설치되어 있는지 확인
- 네트워크 연결 상태 확인

**Q: 게임이 너무 빨리 끝납니다**
- AI 난이도는 `tetris_api_client.py`에서 조정 가능
- 시간 간격을 늘리려면 `time.sleep()` 값 증가 