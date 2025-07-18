import dynamic from 'next/dynamic';

// API 기능이 포함된 테트리스 게임 (hydration 에러 방지)
const TetrisGameWithAPI = dynamic(() => import('../components/TetrisGameWithAPI'), {
  ssr: false,
  loading: () => (
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
  )
});

export default function Home() {
  return (
    <main>
      <TetrisGameWithAPI />
    </main>
  );
} 