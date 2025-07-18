'use client';

import { useEffect, useState } from 'react';
import { useTetris } from '../hooks/useTetris';
import { useGameAPI } from '../hooks/useGameAPI';
import APIMonitor from './APIMonitor';

// 전역 타입 정의
declare global {
  interface Window {
    apiMonitorLog?: (entry: {
      type: 'REQUEST' | 'RESPONSE' | 'ACTION' | 'ERROR';
      method?: string;
      endpoint?: string;
      payload?: any;
      response?: any;
      duration?: number;
      error?: string;
    }) => void;
  }
}

export default function TetrisGameWithAPI() {
  const { gameState, initGame, movePiece, rotatePieceClockwise, hardDrop, togglePause } = useTetris();
  const [isMounted, setIsMounted] = useState(false);
  const [isAPIMode, setIsAPIMode] = useState(false);
  const [showAPIMonitor, setShowAPIMonitor] = useState(false);
  const [monitorWindow, setMonitorWindow] = useState<Window | null>(null);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // AI 모드일 때 키보드 입력 차단
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (isAPIMode) {
        // AI 모드일 때는 모든 키보드 입력 차단 (재시작 R키 제외)
        if (event.key !== 'r' && event.key !== 'R' && event.key !== 'm' && event.key !== 'M') {
          event.preventDefault();
          event.stopPropagation();
        }
      }
      
      // API 모니터 토글 (M 키)
      if (event.key === 'm' || event.key === 'M') {
        event.preventDefault();
        toggleAPIMonitorWindow();
      }
    };

    if (isAPIMode) {
      window.addEventListener('keydown', handleKeyPress, true); // capture phase에서 처리
      return () => window.removeEventListener('keydown', handleKeyPress, true);
    } else {
      // API 모드가 아닐 때도 M 키는 감지
      window.addEventListener('keydown', handleKeyPress);
      return () => window.removeEventListener('keydown', handleKeyPress);
    }
  }, [isAPIMode]);

  // API 액션 핸들러
  const handleAPIAction = (action: string) => {
    if (!isAPIMode) return;
    
    // 렌더링 안정성을 위해 약간의 지연
    setTimeout(() => {
      switch (action) {
        case 'left':
          movePiece('left');
          break;
        case 'right':
          movePiece('right');
          break;
        case 'down':
          movePiece('down');
          break;
        case 'rotate':
          rotatePieceClockwise();
          break;
        case 'drop':
          hardDrop();
          break;
        case 'pause':
          togglePause();
          break;
        case 'restart':
          // AI가 게임 오버를 감지했을 때 재시작 허용 (AI 판단 신뢰)
          console.log(`🔄 AI에서 재시작 요청됨 (현재 상태: ${gameState.isGameOver ? '게임오버' : '진행중'})`);
          initGame();
          break;
      }
    }, 50); // 50ms 지연으로 렌더링 안정성 확보
  };

  // API 훅 사용 (AI 모드일 때만)
  const { isAPIEnabled, consecutiveErrors } = useGameAPI(gameState, handleAPIAction, isAPIMode);

  const renderBoard = () => {
    if (!gameState.board) return null;
    
    const board = gameState.board.map(row => [...row]);
    
    // 현재 피스를 보드에 그리기
    if (gameState.currentPiece) {
      const piece = gameState.currentPiece;
      for (let y = 0; y < piece.shape.length; y++) {
        for (let x = 0; x < piece.shape[y].length; x++) {
          if (piece.shape[y][x]) {
            const boardY = piece.position.y + y;
            const boardX = piece.position.x + x;
            
            if (boardY >= 0 && boardY < 20 && boardX >= 0 && boardX < 10) {
              board[boardY][boardX] = piece.type;
            }
          }
        }
      }
    }

    return board.map((row, rowIndex) => 
      row.map((cell, cellIndex) => (
        <div
          key={`${rowIndex}-${cellIndex}`}
          className={`tetris-cell ${cell ? `filled ${cell}` : ''}`}
        />
      ))
    );
  };

  const handleRestart = () => {
    initGame();
  };

  const toggleAPIMode = () => {
    setIsAPIMode(!isAPIMode);
  };

  const toggleAPIMonitor = () => {
    setShowAPIMonitor(!showAPIMonitor);
  };

  const toggleAPIMonitorWindow = () => {
    if (monitorWindow && !monitorWindow.closed) {
      // 창이 열려있으면 닫기
      monitorWindow.close();
      setMonitorWindow(null);
    } else {
      // 새 창 열기
      const newWindow = window.open(
        '/api-monitor',
        'APIMonitor',
        'width=800,height=600,scrollbars=yes,resizable=yes'
      );
      
      if (newWindow) {
        setMonitorWindow(newWindow);
        
        // 새 창이 준비되면 현재 상태 전송
        const handleMonitorReady = (event: MessageEvent) => {
          if (event.origin !== window.location.origin) return;
          if (event.data.type === 'MONITOR_READY') {
            // 현재 게임 상태를 새 창에 전송
            newWindow.postMessage({
              type: 'GAME_STATE_UPDATE',
              data: { gameState, isAPIMode }
            }, window.location.origin);
            
            window.removeEventListener('message', handleMonitorReady);
          }
        };
        
        window.addEventListener('message', handleMonitorReady);
        
        // 창이 닫히면 상태 업데이트
        const checkClosed = setInterval(() => {
          if (newWindow.closed) {
            setMonitorWindow(null);
            clearInterval(checkClosed);
          }
        }, 1000);
      }
    }
  };

  // 모니터 창에 로그 전송
  const sendLogToMonitor = (logEntry: any) => {
    if (monitorWindow && !monitorWindow.closed) {
      monitorWindow.postMessage({
        type: 'API_LOG',
        data: logEntry
      }, window.location.origin);
    }
  };

  // 게임 상태 변화 시 모니터 창에 업데이트 전송
  useEffect(() => {
    if (monitorWindow && !monitorWindow.closed) {
      monitorWindow.postMessage({
        type: 'GAME_STATE_UPDATE',
        data: { gameState, isAPIMode }
      }, window.location.origin);
    }
  }, [gameState, isAPIMode, monitorWindow]);

  // 전역 API 로깅 함수 설정
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.apiMonitorLog = (entry: any) => {
        // 기존 API 모니터로 전송
        if (showAPIMonitor) {
          // 기존 로직은 APIMonitor 컴포넌트에서 처리
        }
        
        // 별도 창 모니터로 전송
        sendLogToMonitor(entry);
      };
    }
    
    return () => {
      if (typeof window !== 'undefined') {
        delete window.apiMonitorLog;
      }
    };
  }, [showAPIMonitor, sendLogToMonitor]);

  // 페이지 언로드 시 모니터 창 닫기
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (monitorWindow && !monitorWindow.closed) {
        monitorWindow.close();
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      if (monitorWindow && !monitorWindow.closed) {
        monitorWindow.close();
      }
    };
  }, [monitorWindow]);

  // 클라이언트에서만 렌더링
  if (!isMounted) {
    return (
      <div className="game-container">
        <div className="game-info">
          <div className="score">게임 로딩 중...</div>
        </div>
        <div className="tetris-board">
          {Array(200).fill(null).map((_, index) => (
            <div key={index} className="tetris-cell" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="game-container">
      <div className="game-info">
        <div className="score">점수: {gameState.score.toLocaleString()}</div>
        <div className="level">레벨: {gameState.level}</div>
        <div className="level">라인: {gameState.lines}</div>
        
        {/* API 연결 상태 표시 */}
        {isAPIMode && (
          <div style={{ 
            marginTop: '0.5rem', 
            fontSize: '0.75rem', 
            padding: '0.5rem', 
            borderRadius: '4px',
            backgroundColor: isAPIEnabled ? '#dcfce7' : '#fef2f2',
            border: `1px solid ${isAPIEnabled ? '#22c55e' : '#ef4444'}`,
            color: isAPIEnabled ? '#15803d' : '#dc2626'
          }}>
            <div style={{ fontWeight: 'bold' }}>
              {isAPIEnabled ? '🟢 API 연결됨' : '🔴 API 연결 끊김'}
            </div>
            {consecutiveErrors > 0 && (
              <div>⚠️ 연속 에러: {consecutiveErrors}회</div>
            )}
            {!isAPIEnabled && (
              <div>🔄 30초 후 재연결 시도</div>
            )}
          </div>
        )}

        {/* 디버깅: 실시간 상태 정보 */}
        {isAPIMode && (
          <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', color: '#64748b', border: '1px solid #e2e8f0', padding: '0.3rem', borderRadius: '3px' }}>
            <div>🔍 디버그: L{gameState.lines} S{gameState.score}</div>
            <div>📊 상태: {gameState.isGameOver ? '🚨 게임오버' : gameState.isPaused ? '⏸️ 일시정지' : '🎮 진행중'}</div>
            <div>🎯 현재피스: {gameState.currentPiece?.type || 'None'}</div>
            <div>📍 위치: {gameState.currentPiece ? `(${gameState.currentPiece.position.x}, ${gameState.currentPiece.position.y})` : 'None'}</div>
          </div>
        )}
        {gameState.isPaused && <div style={{ color: '#eab308', marginTop: '0.5rem' }}>일시정지</div>}
        
        {/* API 모드 토글 */}
        <div style={{ marginTop: '1rem' }}>
          <button 
            onClick={toggleAPIMode}
            style={{
              background: isAPIMode ? '#22c55e' : '#64748b',
              color: 'white',
              border: 'none',
              padding: '0.5rem 1rem',
              borderRadius: '4px',
              cursor: 'pointer',
              marginRight: '0.5rem'
            }}
          >
            {isAPIMode ? '🤖 AI 모드 ON' : '🎮 수동 모드'}
          </button>
          
          <button 
            onClick={toggleAPIMonitor}
            style={{
              background: showAPIMonitor ? '#3b82f6' : '#64748b',
              color: 'white',
              border: 'none',
              padding: '0.5rem 1rem',
              borderRadius: '4px',
              cursor: 'pointer',
              marginRight: '0.5rem'
            }}
          >
            {showAPIMonitor ? '🔍 인라인 ON' : '🔍 인라인 모니터'}
          </button>
          
          <button 
            onClick={toggleAPIMonitorWindow}
            style={{
              background: (monitorWindow && !monitorWindow.closed) ? '#10b981' : '#64748b',
              color: 'white',
              border: 'none',
              padding: '0.5rem 1rem',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            {(monitorWindow && !monitorWindow.closed) ? '🪟 별도창 ON' : '🪟 별도창 모니터'}
          </button>
        </div>
        
        {/* 디버깅 정보 */}
        <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#64748b' }}>
          <div>M: API 모니터 토글 (별도 창)</div>
          {isAPIMode ? (
            <div>
              <div>🤖 AI 모드: Python AI가 게임을 자동으로 플레이합니다</div>
              {!isAPIEnabled && <div style={{ color: '#ef4444' }}>⚠️ Python AI 서버를 실행해주세요</div>}
            </div>
          ) : (
            <div>🎮 수동 모드: 키보드로 직접 플레이하세요 (←↓→ 스페이스바)</div>
          )}
        </div>
      </div>
      
      <div className="tetris-board">
        {renderBoard()}
      </div>
      
      <div className="controls">
        {isAPIMode ? (
          <>
            <div style={{ color: '#22c55e', fontWeight: 'bold' }}>🤖 AI가 제어 중...</div>
            <div>Python AI가 게임을 플레이합니다</div>
            <div>R : 재시작 (AI/수동 모두 가능)</div>
            <div>M : API 모니터 (별도 창)</div>
          </>
        ) : (
          <>
            <div>← → : 이동</div>
            <div>↓ : 빠른 드롭</div>
            <div>↑ : 회전</div>
            <div>Space : 하드 드롭</div>
            <div>P : 일시정지</div>
            <div>R : 재시작</div>
            <div>M : API 모니터 (별도 창)</div>
          </>
        )}
      </div>

      {gameState.isGameOver && (
        <div className="game-over">
          <div className="game-over-content">
            <h2>게임 오버!</h2>
            <p>최종 점수: {gameState.score.toLocaleString()}</p>
            <p>클리어한 라인: {gameState.lines}</p>
            <button className="restart-btn" onClick={handleRestart}>
              다시 시작
            </button>
            <p style={{ marginTop: '1rem', color: '#64748b', fontSize: '0.9rem' }}>
              또는 R 키를 눌러서 재시작
            </p>
          </div>
        </div>
      )}
      
      {/* API 상태 표시 */}
      <div style={{ 
        position: 'fixed', 
        top: '10px', 
        right: '10px', 
        background: 'rgba(0,0,0,0.8)', 
        color: 'white', 
        padding: '10px',
        fontSize: '12px',
        borderRadius: '5px'
      }}>
        <div>API 모드: {isAPIMode ? 'ON' : 'OFF'}</div>
        <div>현재 피스: {gameState.currentPiece?.type || 'None'}</div>
        <div>위치: {gameState.currentPiece ? `${gameState.currentPiece.position.x}, ${gameState.currentPiece.position.y}` : 'None'}</div>
        <div>게임오버: {gameState.isGameOver ? 'Yes' : 'No'}</div>
        <div style={{ marginTop: '5px', color: '#94a3b8' }}>
          모니터: {(showAPIMonitor || (monitorWindow && !monitorWindow.closed)) ? 'ON' : 'OFF'}
        </div>
      </div>
      
      {/* API 모니터 */}
      <APIMonitor 
        isVisible={showAPIMonitor}
        onClose={() => setShowAPIMonitor(false)}
        isAPIMode={isAPIMode}
        gameState={gameState}
      />
    </div>
  );
} 