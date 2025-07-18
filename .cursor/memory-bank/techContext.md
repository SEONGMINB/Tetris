# 기술 컨텍스트 (Tech Context)

## 🛠️ 기술 스택

### Frontend (웹 게임)
- **프레임워크**: Next.js 14 (App Router)
- **언어**: TypeScript 5.0+
- **UI 라이브러리**: React 18
- **스타일링**: CSS Modules (추정)
- **상태 관리**: React Hooks (useState, useEffect, useCallback)

### Backend (API 서버)
- **서버**: Next.js API Routes
- **언어**: TypeScript/JavaScript
- **데이터 저장**: 메모리 내 상태 (임시)
- **통신**: RESTful API (JSON)

### AI 시스템
- **언어**: Python 3.8+
- **ML 라이브러리**: NumPy (수치 계산)
- **HTTP 클라이언트**: Requests
- **알고리즘**: 커스텀 Q-Learning 구현
- **데이터 저장**: JSON 파일 (모델 영속성)

## 📁 프로젝트 구조

```
/
├── app/                     # Next.js App Router
│   ├── page.tsx            # 메인 페이지
│   ├── layout.tsx          # 레이아웃 컴포넌트
│   └── api/                # API 라우트
│       └── game/           # 게임 API 엔드포인트
├── components/             # React 컴포넌트
│   ├── TetrisGame.tsx     # 메인 게임 컴포넌트
│   ├── TetrisBoard.tsx    # 게임 보드
│   ├── GameStats.tsx      # 점수/레벨 표시
│   └── AIControls.tsx     # AI 모드 컨트롤
├── hooks/                  # 커스텀 React 훅
│   └── useGameAPI.ts      # API 통신 훅
├── types/                  # TypeScript 타입 정의
│   └── tetris.ts          # 게임 상태 타입
├── utils/                  # 유틸리티 함수
│   └── tetris.ts          # 게임 로직 함수
├── python_ai/              # Python AI 시스템
│   ├── reinforcement_tetris_ai.py    # 강화학습 AI
│   ├── advanced_tetris_ai.py         # 휴리스틱 AI
│   ├── tetris_api_client.py          # API 클라이언트
│   ├── requirements.txt              # Python 의존성
│   └── *.json                        # 저장된 모델
├── package.json            # Node.js 의존성
├── tsconfig.json          # TypeScript 설정
└── next.config.js         # Next.js 설정
```

## ⚙️ 개발 환경 설정

### 1. Node.js 환경
```bash
# 필수 도구
Node.js: 18.0+ (LTS 권장)
npm 또는 pnpm
```

### 2. Python 환경
```bash
# Python 패키지
pip install -r python_ai/requirements.txt

# 주요 의존성
selenium==4.15.0        # 웹 자동화 (사용 시)
numpy==1.24.3          # 수치 계산
requests               # HTTP 클라이언트
```

### 3. 실행 환경
```bash
# 웹 서버 실행
npm run dev  # 또는 pnpm dev

# AI 실행
cd python_ai
python reinforcement_tetris_ai.py
```

## 🔧 기술적 제약사항

### 성능 제약
- **폴링 간격**: 200ms (AI 응답성 vs 서버 부하 균형)
- **메모리 사용**: 브라우저 내 게임 상태 저장
- **모델 크기**: JSON 기반 Q-table (확장성 제한)

### 통신 제약
- **동기화 지연**: HTTP 폴링으로 인한 약간의 지연
- **에러 처리**: 네트워크 실패 시 게임 진행 유지
- **단방향 통신**: 웹 → AI 방향이 주요 흐름

### 플랫폼 제약
- **브라우저 호환성**: 모던 브라우저 (ES6+ 지원)
- **Python 버전**: 3.8+ (f-string, 타입 힌트 사용)
- **OS 독립성**: 웹 기반이므로 플랫폼 무관

## 🔌 주요 의존성

### package.json (Node.js)
```json
{
  "dependencies": {
    "next": "14.x",
    "react": "18.x", 
    "react-dom": "18.x",
    "typescript": "5.x"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18"
  }
}
```

### requirements.txt (Python)
```txt
requests>=2.28.0
numpy>=1.24.0
selenium>=4.15.0        # 브라우저 자동화 (선택)
webdriver-manager>=4.0  # 드라이버 관리 (선택)
```

## 🏃‍♂️ 개발 워크플로우

### 1. 로컬 개발
```bash
# 터미널 1: 웹 서버
npm run dev

# 터미널 2: AI 개발/테스트
cd python_ai
python reinforcement_tetris_ai.py
```

### 2. 코드 구조 패턴
- **컴포넌트**: 작은 단위로 분리된 React 컴포넌트
- **훅**: 비즈니스 로직을 커스텀 훅으로 분리
- **타입**: 모든 게임 상태에 TypeScript 타입 적용
- **API**: RESTful 설계 원칙 준수

### 3. 디버깅 전략
```typescript
// 개발 모드에서 상세 로그
if (process.env.NODE_ENV === 'development') {
  console.log('Game state:', gameState);
}
```

```python
# Python AI 디버깅
print(f"🎮 스텝 {step}: 점수 {score}, 액션 {action}")
```

## 🚀 배포 고려사항

### 웹 애플리케이션
- **빌드**: `npm run build` → `.next` 폴더
- **배포**: Vercel, Netlify 등 JAMstack 플랫폼
- **환경 변수**: API URL 설정

### Python AI
- **실행 환경**: 로컬 머신 또는 별도 서버
- **의존성**: 가상 환경 (venv, conda) 권장
- **모델 저장**: JSON 파일 자동 저장/로드

## 🔍 성능 모니터링

### 웹 성능
- **렌더링**: React DevTools로 컴포넌트 렌더링 추적
- **API 응답**: 브라우저 DevTools Network 탭
- **메모리**: 게임 상태 크기 모니터링

### AI 성능
- **학습 속도**: 에피소드별 학습 시간
- **메모리 사용**: Q-table 크기 추적
- **수렴성**: 점수 향상 추이 분석

## 🔧 개발 도구

### IDE/편집기
- **VS Code**: TypeScript, Python 지원
- **확장**: ES7+ React/Redux snippets, Python

### 디버깅 도구
- **브라우저**: Chrome DevTools
- **Python**: print 기반 디버깅, JSON 출력
- **API**: Postman/Insomnia (API 테스트)

### 버전 관리
- **Git**: 코드 버전 관리
- **브랜치**: feature, develop, main 구조

## 🔄 기술 진화 방향

### 단기 개선
- [ ] WebSocket 통신으로 실시간성 향상
- [ ] Redux/Zustand로 상태 관리 개선
- [ ] Python 타입 힌트 강화

### 중기 개선
- [ ] TensorFlow.js로 브라우저 내 AI 구현
- [ ] Docker 컨테이너화
- [ ] CI/CD 파이프라인 구축

### 장기 비전
- [ ] 마이크로서비스 아키텍처
- [ ] 클라우드 AI 서비스 통합
- [ ] 실시간 멀티플레이어 지원 