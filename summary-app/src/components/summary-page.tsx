'use client';

import { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import ReactMarkdown from 'react-markdown';
import { ScrollArea } from '@/components/ui/scroll-area';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm'; // GitHub 스타일 markdown 플러그인 추가
import 'katex/dist/katex.min.css';  // Katex 스타일 추가
import { Button } from "@/components/ui/button";  // Button 컴포넌트 추가
import 'github-markdown-css';  // GitHub 스타일 마크다운 CSS 추가

function cleanMarkdownContent(content: string): string {
  return content.replace(/\n\s*\n/g, '\n'); // 두 줄 이상의 공백을 한 줄로 축소
}

export default function SummaryPage() {
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [summarizedContent, setSummarizedContent] = useState<string>('');
  const [isChatMode, setIsChatMode] = useState<boolean>(false); // Chat 모드 상태 추가
  const [chatMessages, setChatMessages] = useState<{ user: string; bot?: string }[]>([]); // 채팅 메시지 저장
  const [message, setMessage] = useState<string>(''); // 현재 입력된 메시지
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

  // 채팅 모드와 요약 모드 전환
  const toggleChatMode = () => {
    setIsChatMode(!isChatMode);
  };

  // 메시지 전송 함수
  const sendMessage = async () => {
    if (message.trim() === '') return;

    // 사용자 메시지를 추가하고, 봇의 응답을 기다리는 상태로 설정 (Thinking 메시지 추가)
    setChatMessages((prevMessages) => [...prevMessages, { user: `User: ${message}` }]);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: message }),  // 'prompt' 키로 메시지를 보냄
      });

      const data = await response.json();

      if (response.ok) {
        // 봇의 응답을 업데이트하여 상태 변경
        setChatMessages((prevMessages) =>
          prevMessages.map((msg, index) =>
            index === prevMessages.length - 1 ? { ...msg, bot: `Llama3.1: ${data.response}` } : msg
          )
        );
      } else {
        console.error('Error fetching chat response:', data.error);
      }
    } catch (error) {
      console.error('Error sending message:', error);
    }

    setMessage(''); // 메시지 입력란 초기화
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Home 버튼 */}
      <div className="mb-4">
        <Button onClick={goToHome}>Home</Button>
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
                height="566px"
                style={{ border: 'none' }}
                title="PDF Viewer"
              />
            )}
          </div>
        </div>

        {/* 요약 내용 또는 채팅 창 */}
        <div>
          <h2 className="text-2xl font-semibold mb-4">{isChatMode ? 'Chat' : 'Summary'}</h2>
          <ScrollArea className="h-[600px] w-full rounded-md border p-2">
            <div className="w-full h-full" style={{ whiteSpace: 'pre-wrap' }}>
              {/* Summary 모드일 때는 요약 내용 표시 */}
              {!isChatMode ? (
                <ReactMarkdown
                  remarkPlugins={[remarkMath, remarkGfm]}
                  rehypePlugins={[rehypeKatex, rehypeRaw]}
                >
                  {cleanMarkdownContent(summarizedContent)}
                </ReactMarkdown>
              ) : (
                <div className="chat-box">
                  {/* 채팅 창 구현 */}
                  <div className="chat-messages">
                    {chatMessages.map((msg, index) => (
                      <div key={index}>
                        {/* User 메시지 */}
                        {msg.user && (
                          <div className="flex justify-end mb-2">
                            <div className="bg-blue-500 text-white p-2 rounded-md text-right max-w-[60%]">
                              {msg.user}
                            </div>
                          </div>
                        )}
                        {/* Bot 메시지 */}
                        {msg.bot && (
                          <div className="flex justify-start mb-2">
                            <div className="bg-gray-200 p-2 rounded-md text-left max-w-[60%]">
                              {msg.bot}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                  <div className="chat-input mt-4">
                    <input
                      type="text"
                      value={message}
                      onChange={(e) => setMessage(e.target.value)}
                      placeholder="Type your message..."
                      className="border rounded p-2 w-full"
                    />
                    <Button onClick={sendMessage} className="mt-2">Send</Button>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
      </div>

      {/* 우측 하단에 Summary와 Chat 모드를 전환하는 버튼 */}
      <div className="fixed bottom-4 right-4">
        <Button onClick={toggleChatMode}>
          {isChatMode ? 'Summary' : 'Chat'}
        </Button>
      </div>
    </div>
  );
}