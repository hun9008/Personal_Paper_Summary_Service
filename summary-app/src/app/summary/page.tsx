// app/summary/page.tsx
import dynamic from 'next/dynamic'

const SummaryPage = dynamic(() => import('../../components/summary-page'), {
  ssr: false, // 서버 사이드 렌더링을 비활성화하여 클라이언트에서만 실행
})

export default function Page() {
  return <SummaryPage />
}