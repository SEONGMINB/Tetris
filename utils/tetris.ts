import { 
  Tetromino, 
  TetrominoType, 
  Position, 
  GameState, 
  BOARD_WIDTH, 
  BOARD_HEIGHT, 
  TETROMINO_SHAPES, 
  TETROMINO_COLORS 
} from '../types/tetris';

export function createEmptyBoard(): (TetrominoType | null)[][] {
  return Array(BOARD_HEIGHT).fill(null).map(() => Array(BOARD_WIDTH).fill(null));
}

export function createRandomTetromino(): Tetromino {
  const types: TetrominoType[] = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];
  const type = types[Math.floor(Math.random() * types.length)];
  
  return {
    type,
    shape: TETROMINO_SHAPES[type],
    position: { 
      x: Math.floor(BOARD_WIDTH / 2) - Math.floor(TETROMINO_SHAPES[type][0].length / 2), 
      y: -1
    },
    color: TETROMINO_COLORS[type]
  };
}

export function isValidPosition(board: (TetrominoType | null)[][], piece: Tetromino, newPosition: Position): boolean {
  for (let y = 0; y < piece.shape.length; y++) {
    for (let x = 0; x < piece.shape[y].length; x++) {
      if (piece.shape[y][x]) {
        const newX = newPosition.x + x;
        const newY = newPosition.y + y;
        
        if (newX < 0 || newX >= BOARD_WIDTH || newY >= BOARD_HEIGHT) {
          return false;
        }
        
        if (newY >= 0 && board[newY][newX]) {
          return false;
        }
      }
    }
  }
  return true;
}

export function rotatePiece(piece: Tetromino): Tetromino {
  // O 블록은 회전하지 않음
  if (piece.type === 'O') {
    return piece;
  }
  
  const rotated = piece.shape[0].map((_, index) =>
    piece.shape.map(row => row[index]).reverse()
  );
  
  return {
    ...piece,
    shape: rotated
  };
}

export function placePiece(board: (TetrominoType | null)[][], piece: Tetromino): (TetrominoType | null)[][] {
  const newBoard = board.map(row => [...row]);
  
  for (let y = 0; y < piece.shape.length; y++) {
    for (let x = 0; x < piece.shape[y].length; x++) {
      if (piece.shape[y][x]) {
        const boardY = piece.position.y + y;
        const boardX = piece.position.x + x;
        
        if (boardY >= 0) {
          newBoard[boardY][boardX] = piece.type;
        }
      }
    }
  }
  
  return newBoard;
}

export function clearLines(board: (TetrominoType | null)[][]): { board: (TetrominoType | null)[][], linesCleared: number } {
  // 완전히 채워진 줄을 찾기 위한 더 명확한 로직
  const completedLines: number[] = [];
  
  // 완성된 라인 찾기
  for (let row = 0; row < board.length; row++) {
    // 행의 모든 칸이 채워졌는지 확인 (null이 아닌 값으로)
    const isComplete = board[row].every(cell => cell !== null && cell !== undefined);
    if (isComplete) {
      completedLines.push(row);
    }
  }
  
  // 완성된 라인이 없으면 그대로 반환
  if (completedLines.length === 0) {
    return { board, linesCleared: 0 };
  }
  
  // 완성된 라인들을 제거한 새 보드 생성
  const newBoard = board.filter((_, rowIndex) => !completedLines.includes(rowIndex));
  const linesCleared = completedLines.length;
  
  // 제거된 줄 수만큼 위에 빈 줄 추가
  while (newBoard.length < BOARD_HEIGHT) {
    newBoard.unshift(Array(BOARD_WIDTH).fill(null));
  }
  
  // 디버깅을 위한 로그
  if (linesCleared > 0) {
    console.log(`🎯 라인 클리어! ${linesCleared}줄 제거됨 (라인 위치: ${completedLines.join(', ')})`);
  }
  
  return { board: newBoard, linesCleared };
}

export function calculateScore(linesCleared: number, level: number): number {
  const baseScores = [0, 40, 100, 300, 1200];
  return baseScores[linesCleared] * (level + 1);
}

export function getDropInterval(level: number): number {
  return Math.max(300, 1500 - (level * 100));
} 