import { useEffect, useCallback, useRef, useState } from 'react';
import { GameState } from '../types/tetris';

// API 모니터 로깅 함수 타입
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

export function useGameAPI(gameState: GameState, onAction: (action: string) => void, isAPIMode: boolean) {
  const updateTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [consecutiveErrors, setConsecutiveErrors] = useState(0);
  const [isAPIEnabled, setIsAPIEnabled] = useState(true);
  
  // API 로깅 헬퍼 함수
  const logAPI = useCallback((entry: Parameters<NonNullable<Window['apiMonitorLog']>>[0]) => {
    if (typeof window !== 'undefined' && window.apiMonitorLog) {
      window.apiMonitorLog(entry);
    }
  }, []);
  
  // 에러 발생 시 백오프 전략
  const handleAPIError = useCallback((error: string) => {
    setConsecutiveErrors(prev => {
      const newCount = prev + 1;
      
      // 5번 연속 실패 시 API 비활성화
      if (newCount >= 5) {
        setIsAPIEnabled(false);
        console.warn('API 연결이 5번 연속 실패했습니다. API 모드를 일시적으로 비활성화합니다.');
        
        // 30초 후 재시도
        setTimeout(() => {
          console.log('API 연결 재시도...');
          setIsAPIEnabled(true);
          setConsecutiveErrors(0);
        }, 30000);
      }
      
      return newCount;
    });
    
    logAPI({
      type: 'ERROR',
      error: `연속 에러 ${consecutiveErrors + 1}회: ${error}`
    });
  }, [consecutiveErrors, logAPI]);
  
  // 성공 시 에러 카운터 리셋
  const handleAPISuccess = useCallback(() => {
    if (consecutiveErrors > 0) {
      setConsecutiveErrors(0);
      console.log('API 연결이 복구되었습니다.');
    }
  }, [consecutiveErrors]);
  
  // 게임 상태를 API로 전송 (AI 모드일 때만, 디바운싱 적용)
  const updateGameState = useCallback(async (state: GameState) => {
    if (!isAPIMode || !isAPIEnabled) return; // AI 모드가 아니거나 API가 비활성화되면 호출 안함
    
    // 이전 업데이트 타임아웃 취소
    if (updateTimeoutRef.current) {
      clearTimeout(updateTimeoutRef.current);
    }
    
    // 300ms 후에 업데이트 (디바운싱)
    updateTimeoutRef.current = setTimeout(async () => {
      const startTime = Date.now();
      const payload = {
        type: 'update',
        state: {
          board: state.board,
          currentPiece: state.currentPiece,
          nextPiece: state.nextPiece,
          score: state.score,
          level: state.level,
          lines: state.lines,
          isGameOver: state.isGameOver,
          isPaused: state.isPaused
        }
      };
      
      // 요청 로깅
      logAPI({
        type: 'REQUEST',
        method: 'POST',
        endpoint: '/api/game',
        payload: payload
      });
      
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5초 타임아웃
        
        const response = await fetch('/api/game', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        const duration = Date.now() - startTime;
        const responseData = await response.json();
        
        // 응답 로깅
        logAPI({
          type: 'RESPONSE',
          method: 'POST',
          endpoint: '/api/game',
          response: responseData,
          duration: duration
        });
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        handleAPISuccess();
        
      } catch (error) {
        const duration = Date.now() - startTime;
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        
        // 오류 로깅
        logAPI({
          type: 'ERROR',
          method: 'POST',
          endpoint: '/api/game',
          error: `Update failed: ${errorMessage}`,
          duration: duration
        });
        
        handleAPIError(`Update failed: ${errorMessage}`);
      }
    }, 300);
  }, [isAPIMode, isAPIEnabled, logAPI, handleAPIError, handleAPISuccess]);

  // API에서 액션을 가져와서 실행 (AI 모드일 때만)
  const fetchAndExecuteActions = useCallback(async () => {
    if (!isAPIMode || !isAPIEnabled) return; // AI 모드가 아니거나 API가 비활성화되면 액션 폴링 안함
    
    const startTime = Date.now();
    const payload = { type: 'getActions' };
    
    // 요청 로깅
    logAPI({
      type: 'REQUEST',
      method: 'POST',
      endpoint: '/api/game',
      payload: payload
    });
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5초 타임아웃
      
      const response = await fetch('/api/game', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      const duration = Date.now() - startTime;
      const result = await response.json();
      
      // 응답 로깅
      logAPI({
        type: 'RESPONSE',
        method: 'POST',
        endpoint: '/api/game',
        response: result,
        duration: duration
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      handleAPISuccess();
      
      if (result.success && result.actions.length > 0) {
        // 각 액션 실행 로깅
        result.actions.forEach((action: string) => {
          logAPI({
            type: 'ACTION',
            payload: { action: action, executedAt: new Date().toISOString() }
          });
          
          onAction(action);
        });
      }
      
    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // 오류 로깅
      logAPI({
        type: 'ERROR',
        method: 'POST',
        endpoint: '/api/game',
        error: `Fetch actions failed: ${errorMessage}`,
        duration: duration
      });
      
      handleAPIError(`Fetch actions failed: ${errorMessage}`);
    }
  }, [onAction, isAPIMode, isAPIEnabled, logAPI, handleAPIError, handleAPISuccess]);

  // 게임 상태가 변경될 때마다 API 업데이트 (AI 모드일 때만)
  useEffect(() => {
    if (isAPIMode && isAPIEnabled) {
      updateGameState(gameState);
    }
  }, [gameState, updateGameState, isAPIMode, isAPIEnabled]);

  // 주기적으로 액션 확인 (AI 모드일 때만)
  useEffect(() => {
    if (!isAPIMode || !isAPIEnabled) return;
    
    // 연속 에러에 따른 동적 간격 조정
    const intervalMs = consecutiveErrors > 0 ? Math.min(5000, 1000 * Math.pow(2, consecutiveErrors)) : 500;
    
    const interval = setInterval(fetchAndExecuteActions, intervalMs);
    return () => clearInterval(interval);
  }, [fetchAndExecuteActions, isAPIMode, isAPIEnabled, consecutiveErrors]);

  // 컴포넌트 언마운트 시 타임아웃 정리
  useEffect(() => {
    return () => {
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
  }, []);

  return {
    updateGameState,
    fetchAndExecuteActions,
    isAPIEnabled,
    consecutiveErrors
  };
} 