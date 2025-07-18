import './globals.css'

export const metadata = {
  title: 'Tetris Game',
  description: 'Next.js로 만든 테트리스 게임',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  )
} 