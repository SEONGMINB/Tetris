'use client';

import { useState, useEffect, useRef } from 'react';

interface APILogEntry {
  id: string;
  timestamp: string;
  type: 'REQUEST' | 'RESPONSE' | 'ACTION' | 'ERROR';
  method?: string;
  endpoint?: string;
  payload?: any;
  response?: any;
  duration?: number;
  error?: string;
  gameState?: any;
}

export default function APIMonitorPage() {
  const [logs, setLogs] = useState<APILogEntry[]>([]);
  const [filter, setFilter] = useState<'ALL' | 'REQUEST' | 'RESPONSE' | 'ACTION' | 'ERROR'>('ALL');
  const [autoScroll, setAutoScroll] = useState(true);
  const [isAPIMode, setIsAPIMode] = useState(false);
  const [gameState, setGameState] = useState<any>({});
  const logsEndRef = useRef<HTMLDivElement>(null);
  const logCountRef = useRef(0);

  // 부모 창으로부터 메시지 수신
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.origin !== window.location.origin) return;

      const { type, data } = event.data;
      
      switch (type) {
        case 'API_LOG':
          addLog(data);
          break;
        case 'GAME_STATE_UPDATE':
          setGameState(data.gameState);
          setIsAPIMode(data.isAPIMode);
          break;
        case 'CLEAR_LOGS':
          setLogs([]);
          logCountRef.current = 0;
          break;
      }
    };

    window.addEventListener('message', handleMessage);
    
    // 부모 창에 준비 완료 신호 전송
    if (window.opener) {
      window.opener.postMessage({ type: 'MONITOR_READY' }, window.location.origin);
    }

    return () => {
      window.removeEventListener('message', handleMessage);
    };
  }, []);

  // 자동 스크롤
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  const addLog = (entry: Partial<APILogEntry>) => {
    const newLog: APILogEntry = {
      id: `log-${++logCountRef.current}`,
      timestamp: new Date().toLocaleTimeString(),
      type: entry.type || 'ACTION',
      method: entry.method,
      endpoint: entry.endpoint,
      payload: entry.payload,
      response: entry.response,
      duration: entry.duration,
      error: entry.error,
      gameState: entry.gameState
    };

    setLogs(prev => {
      const updated = [...prev, newLog];
      // 최대 500개 로그 유지
      return updated.length > 500 ? updated.slice(-500) : updated;
    });
  };

  const clearLogs = () => {
    setLogs([]);
    logCountRef.current = 0;
  };

  const getLogColor = (type: string) => {
    switch (type) {
      case 'REQUEST': return '#3b82f6';
      case 'RESPONSE': return '#10b981';
      case 'ACTION': return '#f59e0b';
      case 'ERROR': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const formatPayload = (payload: any) => {
    if (!payload) return 'None';
    if (typeof payload === 'string') return payload;
    return JSON.stringify(payload, null, 2);
  };

  const filteredLogs = logs.filter(log => filter === 'ALL' || log.type === filter);

  return (
    <div style={{ 
      padding: '1rem', 
      fontFamily: 'system-ui, -apple-system, sans-serif',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: '#1f2937',
      color: '#f9fafb'
    }}>
      {/* 헤더 */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        padding: '0.5rem 1rem',
        backgroundColor: '#374151',
        borderRadius: '8px',
        marginBottom: '1rem'
      }}>
        <h3 style={{ margin: 0, color: '#f9fafb' }}>🔍 API 모니터</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ 
            padding: '0.25rem 0.5rem',
            borderRadius: '4px',
            backgroundColor: isAPIMode ? '#10b981' : '#ef4444',
            color: 'white',
            fontSize: '0.875rem'
          }}>
            {isAPIMode ? '🟢 API 모드' : '🔴 수동 모드'}
          </span>
          <button 
            onClick={() => window.close()}
            style={{
              background: '#ef4444',
              color: 'white',
              border: 'none',
              padding: '0.25rem 0.5rem',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* 컨트롤 패널 */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        padding: '0.5rem 1rem',
        backgroundColor: '#374151',
        borderRadius: '8px',
        marginBottom: '1rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            필터:
            <select 
              value={filter} 
              onChange={(e) => setFilter(e.target.value as any)}
              style={{
                background: '#4b5563',
                color: 'white',
                border: '1px solid #6b7280',
                padding: '0.25rem',
                borderRadius: '4px'
              }}
            >
              <option value="ALL">전체</option>
              <option value="REQUEST">요청</option>
              <option value="RESPONSE">응답</option>
              <option value="ACTION">액션</option>
              <option value="ERROR">오류</option>
            </select>
          </label>
          
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <input 
              type="checkbox" 
              checked={autoScroll} 
              onChange={(e) => setAutoScroll(e.target.checked)} 
            />
            자동 스크롤
          </label>
        </div>
        
        <button 
          onClick={clearLogs}
          style={{
            background: '#f59e0b',
            color: 'white',
            border: 'none',
            padding: '0.5rem 1rem',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          로그 지우기
        </button>
      </div>

      {/* 통계 */}
      <div style={{ 
        display: 'flex', 
        gap: '1rem',
        padding: '0.5rem 1rem',
        backgroundColor: '#374151',
        borderRadius: '8px',
        marginBottom: '1rem',
        flexWrap: 'wrap'
      }}>
        <div style={{ fontSize: '0.875rem' }}>총 로그: {logs.length}</div>
        <div style={{ fontSize: '0.875rem' }}>요청: {logs.filter(l => l.type === 'REQUEST').length}</div>
        <div style={{ fontSize: '0.875rem' }}>응답: {logs.filter(l => l.type === 'RESPONSE').length}</div>
        <div style={{ fontSize: '0.875rem' }}>액션: {logs.filter(l => l.type === 'ACTION').length}</div>
        <div style={{ fontSize: '0.875rem' }}>오류: {logs.filter(l => l.type === 'ERROR').length}</div>
      </div>

      {/* 로그 리스트 */}
      <div style={{ 
        flex: 1,
        backgroundColor: '#111827',
        borderRadius: '8px',
        padding: '1rem',
        overflow: 'auto'
      }}>
        {filteredLogs.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#6b7280', padding: '2rem' }}>
            로그가 없습니다
          </div>
        ) : (
          filteredLogs.map((log) => (
            <div key={log.id} style={{ 
              marginBottom: '1rem',
              padding: '0.5rem',
              backgroundColor: '#1f2937',
              borderRadius: '4px',
              borderLeft: `4px solid ${getLogColor(log.type)}`
            }}>
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '1rem',
                marginBottom: '0.5rem',
                fontSize: '0.875rem'
              }}>
                <span style={{ color: '#9ca3af' }}>{log.timestamp}</span>
                <span style={{ 
                  color: getLogColor(log.type),
                  fontWeight: 'bold'
                }}>
                  {log.type}
                </span>
                {log.method && <span style={{ color: '#60a5fa' }}>{log.method}</span>}
                {log.endpoint && <span style={{ color: '#34d399' }}>{log.endpoint}</span>}
                {log.duration && <span style={{ color: '#fbbf24' }}>{log.duration}ms</span>}
              </div>
              
              {log.payload && (
                <div style={{ marginBottom: '0.5rem' }}>
                  <strong style={{ color: '#f3f4f6' }}>Payload:</strong>
                  <pre style={{ 
                    backgroundColor: '#0f172a',
                    padding: '0.5rem',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    overflow: 'auto',
                    color: '#e2e8f0'
                  }}>
                    {formatPayload(log.payload)}
                  </pre>
                </div>
              )}
              
              {log.response && (
                <div style={{ marginBottom: '0.5rem' }}>
                  <strong style={{ color: '#f3f4f6' }}>Response:</strong>
                  <pre style={{ 
                    backgroundColor: '#0f172a',
                    padding: '0.5rem',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    overflow: 'auto',
                    color: '#e2e8f0'
                  }}>
                    {formatPayload(log.response)}
                  </pre>
                </div>
              )}
              
              {log.error && (
                <div style={{ color: '#ef4444' }}>
                  <strong>Error:</strong> {log.error}
                </div>
              )}
            </div>
          ))
        )}
        <div ref={logsEndRef} />
      </div>

      {/* 게임 상태 정보 */}
      {gameState && (
        <div style={{ 
          marginTop: '1rem',
          padding: '0.5rem 1rem',
          backgroundColor: '#374151',
          borderRadius: '8px',
          fontSize: '0.875rem'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>🎮 게임 상태</div>
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <div>점수: {gameState.score?.toLocaleString() || 0}</div>
            <div>라인: {gameState.lines || 0}</div>
            <div>레벨: {gameState.level || 1}</div>
            <div>현재 피스: {gameState.currentPiece?.type || 'None'}</div>
            <div>상태: {gameState.isGameOver ? '🚨 게임오버' : gameState.isPaused ? '⏸️ 일시정지' : '🎮 진행중'}</div>
          </div>
        </div>
      )}
    </div>
  );
} 