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

interface APIMonitorProps {
  isVisible: boolean;
  onClose: () => void;
  isAPIMode: boolean;
  gameState: any;
}

export default function APIMonitor({ isVisible, onClose, isAPIMode, gameState }: APIMonitorProps) {
  const [logs, setLogs] = useState<APILogEntry[]>([]);
  const [filter, setFilter] = useState<'ALL' | 'REQUEST' | 'RESPONSE' | 'ACTION' | 'ERROR'>('ALL');
  const [autoScroll, setAutoScroll] = useState(true);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const logCountRef = useRef(0);

  // 자동 스크롤
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  // 게임 상태 변화 추적
  useEffect(() => {
    if (isAPIMode) {
      addLog({
        type: 'ACTION',
        payload: {
          currentPiece: gameState.currentPiece,
          score: gameState.score,
          lines: gameState.lines,
          isGameOver: gameState.isGameOver,
          isPaused: gameState.isPaused
        },
        gameState: gameState
      });
    }
  }, [gameState, isAPIMode]);

  const addLog = (entry: Partial<APILogEntry>) => {
    const newEntry: APILogEntry = {
      id: `log-${logCountRef.current++}`,
      timestamp: new Date().toLocaleTimeString('ko-KR', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit'
      }) + '.' + new Date().getMilliseconds().toString().padStart(3, '0'),
      type: entry.type || 'ACTION',
      ...entry
    };

    setLogs(prev => {
      const updated = [...prev, newEntry];
      // 최대 500개 로그만 유지
      return updated.slice(-500);
    });
  };

  // 외부에서 로그 추가할 수 있도록 전역 함수 등록
  useEffect(() => {
    if (typeof window !== 'undefined') {
      (window as any).apiMonitorLog = addLog;
    }
    return () => {
      if (typeof window !== 'undefined') {
        delete (window as any).apiMonitorLog;
      }
    };
  }, []);

  const clearLogs = () => {
    setLogs([]);
    logCountRef.current = 0;
  };

  const filteredLogs = logs.filter(log => filter === 'ALL' || log.type === filter);

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

  if (!isVisible) return null;

  return (
    <div className="api-monitor-overlay">
      <div className="api-monitor-window">
        {/* 헤더 */}
        <div className="api-monitor-header">
          <h3>🔍 API 모니터</h3>
          <div className="header-controls">
            <span className={`status-indicator ${isAPIMode ? 'active' : 'inactive'}`}>
              {isAPIMode ? '🟢 API 모드' : '🔴 수동 모드'}
            </span>
            <button onClick={onClose} className="close-btn">✕</button>
          </div>
        </div>

        {/* 컨트롤 패널 */}
        <div className="control-panel">
          <div className="filter-controls">
            <label>필터:</label>
            <select value={filter} onChange={(e) => setFilter(e.target.value as any)}>
              <option value="ALL">전체</option>
              <option value="REQUEST">요청</option>
              <option value="RESPONSE">응답</option>
              <option value="ACTION">액션</option>
              <option value="ERROR">오류</option>
            </select>
          </div>
          
          <div className="action-controls">
            <label>
              <input 
                type="checkbox" 
                checked={autoScroll} 
                onChange={(e) => setAutoScroll(e.target.checked)} 
              />
              자동 스크롤
            </label>
            <button onClick={clearLogs} className="clear-btn">로그 지우기</button>
          </div>
        </div>

        {/* 통계 */}
        <div className="stats-panel">
          <div className="stat-item">
            <span>총 로그: {logs.length}</span>
          </div>
          <div className="stat-item">
            <span>요청: {logs.filter(l => l.type === 'REQUEST').length}</span>
          </div>
          <div className="stat-item">
            <span>응답: {logs.filter(l => l.type === 'RESPONSE').length}</span>
          </div>
          <div className="stat-item">
            <span>액션: {logs.filter(l => l.type === 'ACTION').length}</span>
          </div>
          <div className="stat-item">
            <span>오류: {logs.filter(l => l.type === 'ERROR').length}</span>
          </div>
        </div>

        {/* 로그 리스트 */}
        <div className="logs-container">
          {filteredLogs.length === 0 ? (
            <div className="no-logs">로그가 없습니다</div>
          ) : (
            filteredLogs.map((log) => (
              <div key={log.id} className="log-entry">
                <div className="log-header">
                  <span className="timestamp">{log.timestamp}</span>
                  <span 
                    className="log-type" 
                    style={{ color: getLogColor(log.type) }}
                  >
                    {log.type}
                  </span>
                  {log.method && <span className="method">{log.method}</span>}
                  {log.endpoint && <span className="endpoint">{log.endpoint}</span>}
                  {log.duration && <span className="duration">{log.duration}ms</span>}
                </div>
                
                {log.payload && (
                  <div className="log-payload">
                    <strong>Payload:</strong>
                    <pre>{formatPayload(log.payload)}</pre>
                  </div>
                )}
                
                {log.response && (
                  <div className="log-response">
                    <strong>Response:</strong>
                    <pre>{formatPayload(log.response)}</pre>
                  </div>
                )}
                
                {log.error && (
                  <div className="log-error">
                    <strong>Error:</strong>
                    <span style={{ color: '#ef4444' }}>{log.error}</span>
                  </div>
                )}
                
                {log.gameState && (
                  <div className="log-gamestate">
                    <strong>게임 상태:</strong>
                    <div className="gamestate-summary">
                      <span>점수: {log.gameState.score}</span>
                      <span>라인: {log.gameState.lines}</span>
                      <span>피스: {log.gameState.currentPiece?.type || 'None'}</span>
                      <span>위치: {log.gameState.currentPiece ? 
                        `(${log.gameState.currentPiece.position.x}, ${log.gameState.currentPiece.position.y})` : 
                        'None'}</span>
                      <span>게임오버: {log.gameState.isGameOver ? 'Yes' : 'No'}</span>
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
          <div ref={logsEndRef} />
        </div>
      </div>
      
      <style jsx>{`
        .api-monitor-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.7);
          display: flex;
          justify-content: center;
          align-items: center;
          z-index: 1000;
        }
        
        .api-monitor-window {
          background: #1f2937;
          color: white;
          border-radius: 8px;
          width: 90vw;
          height: 90vh;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }
        
        .api-monitor-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1rem;
          border-bottom: 1px solid #374151;
          background: #111827;
        }
        
        .header-controls {
          display: flex;
          align-items: center;
          gap: 1rem;
        }
        
        .status-indicator {
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.875rem;
          font-weight: bold;
        }
        
        .status-indicator.active {
          background: #065f46;
          color: #10b981;
        }
        
        .status-indicator.inactive {
          background: #7f1d1d;
          color: #f87171;
        }
        
        .close-btn {
          background: #ef4444;
          color: white;
          border: none;
          border-radius: 4px;
          padding: 0.5rem;
          cursor: pointer;
          font-size: 1rem;
        }
        
        .close-btn:hover {
          background: #dc2626;
        }
        
        .control-panel {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.75rem 1rem;
          background: #111827;
          border-bottom: 1px solid #374151;
          gap: 1rem;
        }
        
        .filter-controls, .action-controls {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        
        .filter-controls select, .clear-btn {
          background: #374151;
          color: white;
          border: 1px solid #4b5563;
          border-radius: 4px;
          padding: 0.25rem 0.5rem;
        }
        
        .clear-btn {
          cursor: pointer;
          background: #dc2626;
        }
        
        .clear-btn:hover {
          background: #b91c1c;
        }
        
        .stats-panel {
          display: flex;
          gap: 1rem;
          padding: 0.5rem 1rem;
          background: #0f172a;
          border-bottom: 1px solid #374151;
          font-size: 0.875rem;
        }
        
        .stat-item {
          color: #94a3b8;
        }
        
        .logs-container {
          flex: 1;
          overflow-y: auto;
          padding: 0.5rem;
        }
        
        .no-logs {
          text-align: center;
          color: #6b7280;
          padding: 2rem;
        }
        
        .log-entry {
          border: 1px solid #374151;
          border-radius: 4px;
          margin-bottom: 0.5rem;
          background: #111827;
        }
        
        .log-header {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.5rem;
          background: #1f2937;
          border-bottom: 1px solid #374151;
          font-size: 0.875rem;
        }
        
        .timestamp {
          color: #9ca3af;
          font-family: monospace;
        }
        
        .log-type {
          font-weight: bold;
          font-size: 0.75rem;
          padding: 0.125rem 0.375rem;
          border-radius: 3px;
          background: rgba(0, 0, 0, 0.3);
        }
        
        .method, .endpoint {
          color: #60a5fa;
          font-family: monospace;
        }
        
        .duration {
          color: #fbbf24;
          font-family: monospace;
        }
        
        .log-payload, .log-response, .log-error, .log-gamestate {
          padding: 0.5rem;
          border-top: 1px solid #374151;
        }
        
        .log-payload pre, .log-response pre {
          background: #0f172a;
          padding: 0.5rem;
          border-radius: 3px;
          overflow-x: auto;
          font-size: 0.75rem;
          margin: 0.25rem 0 0 0;
        }
        
        .gamestate-summary {
          display: flex;
          gap: 1rem;
          margin-top: 0.25rem;
          font-size: 0.75rem;
          color: #94a3b8;
        }
        
        .log-error span {
          margin-left: 0.5rem;
        }
      `}</style>
    </div>
  );
} 