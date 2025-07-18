import { NextRequest, NextResponse } from 'next/server';

// 게임 상태를 저장할 간단한 메모리 스토어
let gameState = {
  board: Array(20).fill(null).map(() => Array(10).fill(null)),
  currentPiece: null,
  nextPiece: null,
  score: 0,
  level: 0,
  lines: 0,
  isGameOver: false,
  isPaused: false,
  lastUpdate: Date.now()
};

let pendingActions: string[] = [];

export async function GET() {
  return NextResponse.json({
    success: true,
    data: {
      ...gameState,
      pendingActions: [...pendingActions]
    }
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    if (body.type === 'action') {
      // Python에서 보낸 액션을 큐에 추가
      pendingActions.push(body.action);
      return NextResponse.json({ success: true, message: 'Action queued' });
    }
    
    if (body.type === 'update') {
      // 게임 상태 업데이트 (프론트엔드에서)
      gameState = {
        ...gameState,
        ...body.state,
        lastUpdate: Date.now()
      };
      return NextResponse.json({ success: true, message: 'State updated' });
    }
    
    if (body.type === 'getActions') {
      // 대기 중인 액션들을 반환하고 큐 비우기
      const actions = [...pendingActions];
      pendingActions = [];
      return NextResponse.json({ success: true, actions });
    }
    
    return NextResponse.json({ success: false, message: 'Invalid request type' });
  } catch (error) {
    return NextResponse.json({ success: false, error: 'Invalid JSON' }, { status: 400 });
  }
} 