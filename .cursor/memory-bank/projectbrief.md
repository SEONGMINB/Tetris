# 프로젝트 개요 (Project Brief)

## 🎮 프로젝트 정의
**테트리스 웹 게임 + AI 플레이어 시스템**

웹 브라우저에서 플레이 가능한 클래식 테트리스 게임과 Python 기반 강화학습 AI가 자동으로 게임을 플레이하는 통합 시스템

## 🎯 핵심 목표
1. **완전한 테트리스 게임 구현**
   - 클래식 테트리스 규칙 준수
   - 반응형 웹 인터페이스
   - 실시간 게임 플레이

2. **AI 플레이어 시스템**
   - Python 기반 강화학습 AI
   - 웹 게임과 API 통신
   - 자동 게임 플레이 및 학습

3. **API 기반 통합**
   - 게임 상태 실시간 동기화
   - AI 액션 전달 시스템
   - 브라우저-Python 간 통신

## 📋 주요 기능 요구사항

### 웹 게임 (Frontend)
- [x] 테트리스 보드 (20x10 그리드)
- [x] 7가지 테트로미노 블록
- [x] 블록 회전 및 이동
- [x] 라인 클리어 시스템
- [x] 점수 및 레벨 시스템
- [x] 게임 오버 감지
- [x] 키보드 컨트롤
- [x] AI 모드 토글

### Python AI
- [x] 강화학습 (Q-Learning) 구현
- [x] 게임 상태 분석
- [x] 휴리스틱 평가 함수
- [x] 모델 저장/로드
- [x] 자동 훈련 시스템

### API 시스템
- [x] 게임 상태 전송
- [x] AI 액션 수신
- [x] 실시간 동기화

## 🛠️ 기술 스택
- **Frontend**: Next.js, TypeScript, React
- **Backend**: Next.js API Routes
- **AI**: Python, NumPy, Requests
- **통신**: REST API

## 📁 프로젝트 구조
```
/
├── app/                    # Next.js 앱 디렉토리
├── components/             # React 컴포넌트
├── hooks/                  # 커스텀 훅
├── types/                  # TypeScript 타입
├── utils/                  # 유틸리티 함수
└── python_ai/              # Python AI 시스템
    ├── reinforcement_tetris_ai.py
    ├── advanced_tetris_ai.py
    └── requirements.txt
```

## 🚀 사용 시나리오
1. **수동 플레이**: 사용자가 직접 키보드로 테트리스 플레이
2. **AI 관찰**: AI가 자동으로 플레이하는 모습 관찰
3. **AI 훈련**: 강화학습을 통한 AI 성능 향상
4. **성능 분석**: AI의 학습 진행도 및 성능 평가 