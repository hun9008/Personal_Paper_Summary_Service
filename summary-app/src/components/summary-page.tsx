'use client';

import { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import ReactMarkdown from 'react-markdown';
import { ScrollArea } from '@/components/ui/scroll-area';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
// import remarkGfm from 'remark-gfm'; // GitHub 스타일 markdown 플러그인 추가
import remarkBreaks from 'remark-breaks';  // 줄바꿈 처리 플러그인 추가
import 'katex/dist/katex.min.css';  // Katex 스타일 추가
import { Button } from "@/components/ui/button";  // Button 컴포넌트 추가
// import 'github-markdown-css';  // GitHub 스타일 마크다운 CSS 추가

export default function SummaryPage() {
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [summarizedContent, setSummarizedContent] = useState<string>('');
  const searchParams = useSearchParams();
  const fileName = searchParams.get('file');
  const router = useRouter();  // useRouter 훅 사용

  useEffect(() => {
    async function fetchSummary() {
      if (fileName) {
        try {
          const response = await fetch(`http://localhost:8000/get-summary?file_name=${fileName}`);
          const data = await response.json();
          if (response.ok) {
            setPdfUrl(data.pdf_url);  // 백엔드에서 반환된 PDF URL 사용
            setSummarizedContent(data.summary_content);  // 요약 내용 설정
          } else {
            console.error('Error fetching summary:', data.error);
          }
        } catch (error) {
          console.error('Error fetching summary:', error);
        }
      }
    }
    fetchSummary();
  }, [fileName]);

  // Home으로 이동하는 함수
  const goToHome = () => {
    router.push('/');
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Home 버튼 */}
      <div className="mb-4">
        <Button onClick={goToHome}>Home</Button>  {/* Home 버튼 추가 */}
      </div>

      <h1 className="text-3xl font-bold text-center mb-8">Paper Summary</h1>
      <div className="grid lg:grid-cols-2 gap-8">
        {/* PDF 뷰어 */}
        <div>
          <h2 className="text-2xl font-semibold mb-4">Original Paper</h2>
          <div className="border rounded-md p-4">
            {pdfUrl && (
              <iframe
                src={pdfUrl}
                width="100%"
                height="600px"
                style={{ border: 'none' }}
                title="PDF Viewer"
              />
            )}
          </div>
        </div>

        {/* 요약 내용 */}
        <div>
          <h2 className="text-2xl font-semibold mb-4">Summary</h2>
          <ScrollArea className="h-[600px] w-full rounded-md border overflow-auto p-2 markdown-body">
            <ReactMarkdown
              remarkPlugins={[remarkMath, remarkBreaks]}  // GitHub markdown, 수식 및 줄바꿈 플러그인 추가
              rehypePlugins={[rehypeKatex, rehypeRaw]} // LaTeX와 HTML 렌더링
              components={{
                span: (props) => <span className="red-text" {...props} /> // node 제거
              }}
            >
              {summarizedContent}
            </ReactMarkdown>
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}