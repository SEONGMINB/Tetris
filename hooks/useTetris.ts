import { useState, useEffect, useCallback } from 'react';
import { GameState, Tetromino, Position, BOARD_WIDTH, BOARD_HEIGHT } from '../types/tetris';
import {
  createEmptyBoard,
  createRandomTetromino,
  isValidPosition,
  rotatePiece,
  placePiece,
  clearLines,
  calculateScore,
  getDropInterval
} from '../utils/tetris';

export function useTetris() {
  const [gameState, setGameState] = useState<GameState>(() => ({
    board: createEmptyBoard(),
    currentPiece: null,
    nextPiece: null,
    score: 0,
    level: 0,
    lines: 0,
    isGameOver: false,
    isPaused: false
  }));

  // 클라이언트에서 게임 초기화
  useEffect(() => {
    if (!gameState.currentPiece && !gameState.nextPiece) {
      const currentPiece = createRandomTetromino();
      const nextPiece = createRandomTetromino();
      
      setGameState(prev => ({
        ...prev,
        currentPiece,
        nextPiece
      }));
    }
  }, [gameState.currentPiece, gameState.nextPiece]);

  const initGame = useCallback(() => {
    console.log("🎮 게임 초기화/재시작!");
    const currentPiece = createRandomTetromino();
    const nextPiece = createRandomTetromino();
    
    setGameState({
      board: createEmptyBoard(),
      currentPiece,
      nextPiece,
      score: 0,
      level: 0,
      lines: 0,
      isGameOver: false,
      isPaused: false
    });
  }, []);

  const movePiece = useCallback((direction: 'left' | 'right' | 'down') => {
    setGameState(prevState => {
      if (!prevState.currentPiece || prevState.isGameOver || prevState.isPaused) {
        return prevState;
      }

      const piece = prevState.currentPiece;
      let newPosition: Position;

      switch (direction) {
        case 'left':
          newPosition = { x: piece.position.x - 1, y: piece.position.y };
          break;
        case 'right':
          newPosition = { x: piece.position.x + 1, y: piece.position.y };
          break;
        case 'down':
          newPosition = { x: piece.position.x, y: piece.position.y + 1 };
          break;
        default:
          return prevState;
      }

      if (isValidPosition(prevState.board, piece, newPosition)) {
        return {
          ...prevState,
          currentPiece: {
            ...piece,
            position: newPosition
          }
        };
      }

      // 아래로 이동할 수 없으면 피스를 고정
      if (direction === 'down') {
        const newBoard = placePiece(prevState.board, piece);
        const { board: clearedBoard, linesCleared } = clearLines(newBoard);
        const scoreToAdd = calculateScore(linesCleared, prevState.level);
        const newLines = prevState.lines + linesCleared;
        const newLevel = Math.floor(newLines / 10);

        // 디버깅: 라인 클리어 상황 로그
        if (linesCleared > 0) {
          console.log(`📊 라인 업데이트: ${prevState.lines} → ${newLines} (+${linesCleared})`);
          console.log(`💯 점수 업데이트: ${prevState.score} → ${prevState.score + scoreToAdd} (+${scoreToAdd})`);
        }

        const nextPiece = createRandomTetromino();
        
        // 게임 오버 체크 - 더 간단하고 확실한 방법
        const nextPieceToCheck = prevState.nextPiece!;
        let isGameOver = false;
        
        // nextPiece가 초기 위치에서 충돌하는지 확인
        if (!isValidPosition(clearedBoard, nextPieceToCheck, nextPieceToCheck.position)) {
          isGameOver = true;
          console.log("🚨 게임 오버 감지: 새 피스가 배치될 수 없음");
        }
        
        // 추가 안전 체크: 상단 3줄에 블록이 있는지 확인
        if (!isGameOver) {
          for (let row = 0; row < 3; row++) {
            for (let col = 0; col < BOARD_WIDTH; col++) {
              if (clearedBoard[row][col] !== null) {
                isGameOver = true;
                console.log(`🚨 게임 오버 감지: 상단 영역에 블록 감지 (${row}, ${col})`);
                break;
              }
            }
            if (isGameOver) break;
          }
        }
        
        if (isGameOver) {
          console.log("🚨 게임 오버 처리됨!");
          console.log(`최종 점수: ${prevState.score + scoreToAdd}, 라인: ${newLines}`);
          return {
            ...prevState,
            board: clearedBoard,
            score: prevState.score + scoreToAdd,
            lines: newLines,
            level: newLevel,
            isGameOver: true
          };
        }

        return {
          ...prevState,
          board: clearedBoard,
          currentPiece: prevState.nextPiece,
          nextPiece: nextPiece,
          score: prevState.score + scoreToAdd,
          lines: newLines,
          level: newLevel
        };
      }

      return prevState;
    });
  }, []);

  const rotatePieceClockwise = useCallback(() => {
    setGameState(prevState => {
      if (!prevState.currentPiece || prevState.isGameOver || prevState.isPaused) {
        return prevState;
      }

      const rotatedPiece = rotatePiece(prevState.currentPiece);
      
      // 기본 위치에서 회전 가능한지 확인
      if (isValidPosition(prevState.board, rotatedPiece, rotatedPiece.position)) {
        return {
          ...prevState,
          currentPiece: rotatedPiece
        };
      }

      // 벽 킥 시도 (왼쪽, 오른쪽으로 한 칸씩 이동해서 회전 가능한지 확인)
      const kickPositions = [
        { x: rotatedPiece.position.x - 1, y: rotatedPiece.position.y },
        { x: rotatedPiece.position.x + 1, y: rotatedPiece.position.y },
        { x: rotatedPiece.position.x, y: rotatedPiece.position.y - 1 }
      ];

      for (const kickPos of kickPositions) {
        if (isValidPosition(prevState.board, rotatedPiece, kickPos)) {
          return {
            ...prevState,
            currentPiece: {
              ...rotatedPiece,
              position: kickPos
            }
          };
        }
      }

      return prevState;
    });
  }, []);

  const hardDrop = useCallback(() => {
    setGameState(prevState => {
      if (!prevState.currentPiece || prevState.isGameOver || prevState.isPaused) {
        return prevState;
      }

      let piece = prevState.currentPiece;
      let newPosition = { ...piece.position };

      // 가능한 한 아래로 이동
      while (isValidPosition(prevState.board, piece, { x: newPosition.x, y: newPosition.y + 1 })) {
        newPosition.y++;
      }

      piece = { ...piece, position: newPosition };
      
      const newBoard = placePiece(prevState.board, piece);
      const { board: clearedBoard, linesCleared } = clearLines(newBoard);
      const scoreToAdd = calculateScore(linesCleared, prevState.level);
      const newLines = prevState.lines + linesCleared;
      const newLevel = Math.floor(newLines / 10);

      // 디버깅: 라인 클리어 상황 로그
      if (linesCleared > 0) {
        console.log(`📊 [하드드롭] 라인 업데이트: ${prevState.lines} → ${newLines} (+${linesCleared})`);
        console.log(`💯 [하드드롭] 점수 업데이트: ${prevState.score} → ${prevState.score + scoreToAdd} (+${scoreToAdd})`);
      }

      const nextPiece = createRandomTetromino();
      
      // 게임 오버 체크 - 더 간단하고 확실한 방법
      const nextPieceToCheck = prevState.nextPiece!;
      let isGameOver = false;
      
      // nextPiece가 초기 위치에서 충돌하는지 확인
      if (!isValidPosition(clearedBoard, nextPieceToCheck, nextPieceToCheck.position)) {
        isGameOver = true;
        console.log("🚨 [하드드롭] 게임 오버 감지: 새 피스가 배치될 수 없음");
      }
      
      // 추가 안전 체크: 상단 3줄에 블록이 있는지 확인
      if (!isGameOver) {
        for (let row = 0; row < 3; row++) {
          for (let col = 0; col < BOARD_WIDTH; col++) {
            if (clearedBoard[row][col] !== null) {
              isGameOver = true;
              console.log(`🚨 [하드드롭] 게임 오버 감지: 상단 영역에 블록 감지 (${row}, ${col})`);
              break;
            }
          }
          if (isGameOver) break;
        }
      }
      
      if (isGameOver) {
        console.log("🚨 [하드드롭] 게임 오버 처리됨!");
        console.log(`최종 점수: ${prevState.score + scoreToAdd}, 라인: ${newLines}`);
        return {
          ...prevState,
          board: clearedBoard,
          score: prevState.score + scoreToAdd,
          lines: newLines,
          level: newLevel,
          isGameOver: true
        };
      }

      return {
        ...prevState,
        board: clearedBoard,
        currentPiece: prevState.nextPiece,
        nextPiece: nextPiece,
        score: prevState.score + scoreToAdd,
        lines: newLines,
        level: newLevel
      };
    });
  }, []);

  const togglePause = useCallback(() => {
    setGameState(prevState => ({
      ...prevState,
      isPaused: !prevState.isPaused
    }));
  }, []);

  // 자동 드롭
  useEffect(() => {
    if (gameState.isGameOver || gameState.isPaused || !gameState.currentPiece) {
      return;
    }

    const interval = setInterval(() => {
      movePiece('down');
    }, getDropInterval(gameState.level));

    return () => clearInterval(interval);
  }, [gameState.level, gameState.isGameOver, gameState.isPaused, gameState.currentPiece, movePiece]);

  // 키보드 이벤트
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      // 게임 오버 상태에서만 R 키로 재시작 허용
      if (gameState.isGameOver) {
        if (event.key === 'r' || event.key === 'R') {
          event.preventDefault();
          initGame();
        }
        return; // 게임 오버 상태에서는 다른 키 입력 무시
      }

      switch (event.key) {
        case 'ArrowLeft':
          event.preventDefault();
          movePiece('left');
          break;
        case 'ArrowRight':
          event.preventDefault();
          movePiece('right');
          break;
        case 'ArrowDown':
          event.preventDefault();
          movePiece('down');
          break;
        case 'ArrowUp':
          event.preventDefault();
          rotatePieceClockwise();
          break;
        case ' ':
          event.preventDefault();
          hardDrop();
          break;
        case 'p':
        case 'P':
          event.preventDefault();
          togglePause();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [gameState.isGameOver, movePiece, rotatePieceClockwise, hardDrop, togglePause, initGame]);

  return {
    gameState,
    initGame,
    movePiece,
    rotatePieceClockwise,
    hardDrop,
    togglePause
  };
} 