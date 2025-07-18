'use client';

import { useEffect } from 'react';
import { useTetris } from '../hooks/useTetris';

export default function TetrisGame() {
  const { gameState, initGame } = useTetris();

  const renderBoard = () => {
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

  return (
    <div className="game-container">
      <div className="game-info">
        <div className="score">점수: {gameState.score.toLocaleString()}</div>
        <div className="level">레벨: {gameState.level}</div>
        <div className="level">라인: {gameState.lines}</div>
        {gameState.isPaused && <div style={{ color: '#eab308', marginTop: '0.5rem' }}>일시정지</div>}
      </div>
      
      <div className="tetris-board">
        {renderBoard()}
      </div>
      
      <div className="controls">
        <div>← → : 이동</div>
        <div>↓ : 빠른 드롭</div>
        <div>↑ : 회전</div>
        <div>Space : 하드 드롭</div>
        <div>P : 일시정지</div>
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
          </div>
        </div>
      )}
      
      {/* 디버그 정보 */}
      {process.env.NODE_ENV === 'development' && (
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
          <div>현재 피스: {gameState.currentPiece?.type || 'None'}</div>
          <div>위치: {gameState.currentPiece ? `${gameState.currentPiece.position.x}, ${gameState.currentPiece.position.y}` : 'None'}</div>
          <div>게임오버: {gameState.isGameOver ? 'Yes' : 'No'}</div>
        </div>
      )}
    </div>
  );
} 