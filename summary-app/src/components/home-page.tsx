'use client';

import { useEffect, useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { UploadIcon, FileTextIcon } from 'lucide-react';
import { useRouter } from 'next/navigation';  // next.js 라우터 사용

export function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);  // 로딩 상태 관리
  const [progress, setProgress] = useState<number>(0);  // 진행 상태 관리
  const [previousUploads, setPreviousUploads] = useState<{ id: number, name: string }[]>([]);  // 이전 업로드 기록

  const router = useRouter();  // useRouter 훅 사용

  // API 호출로 이전 업로드 기록 가져오기
  useEffect(() => {
    async function fetchPreviousUploads() {
      try {
        const response = await fetch('http://localhost:8000/get-prev-summary');
        const data = await response.json();
        if (response.ok) {
          const uploads = data.completed_summaries.map((fileName: string, index: number) => ({
            id: index + 1,
            name: fileName + '.pdf',  // PDF 파일 이름으로 설정
          }));
          setPreviousUploads(uploads);  // 상태 업데이트
        } else {
          console.error('Error fetching previous uploads:', data.error);
        }
      } catch (error) {
        console.error('Error fetching previous uploads:', error);
      }
    }

    fetchPreviousUploads();
  }, []);

  // SSE를 통해 진행 상태를 받는 함수
  const startProgress = () => {
    const eventSource = new EventSource('http://localhost:8000/progress');
    eventSource.onmessage = (event) => {
      const progressValue = parseFloat(event.data);
      if (!isNaN(progressValue)) {
        setProgress(progressValue);  // 진행 상태 업데이트
      } else {
        console.error('Invalid progress value:', event.data);
      }

      if (progressValue >= 100) {
        eventSource.close();  // 100%가 되면 연결 종료
      }
    };
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFile(event.target.files[0]);
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (file) {
      setIsLoading(true);  // 업로드 시작 시 로딩 상태로 전환
      const formData = new FormData();
      formData.append('file', file);

      try {
        startProgress();  // 진행 상태 모니터링 시작

        const response = await fetch('http://localhost:8000/upload/', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          console.log('Upload successful:', data);

          // 파일명을 요약 페이지로 전달
          const fileName = file.name.replace('.pdf', ''); // 파일명에서 확장자를 제거
          router.push(`/summary?file=${fileName}`);  // summary 페이지로 파일명 전달
        } else {
          console.error('Upload failed:', response.statusText);
        }
      } catch (error) {
        console.error('Error uploading file:', error);
      } finally {
        setIsLoading(false);  // 업로드 완료 후 로딩 상태 해제
      }
    }
  };

  // 이전 업로드된 논문 클릭 시 처리
  const handlePreviousUploadClick = (fileName: string) => {
    // 해당 파일명으로 summary 페이지로 이동
    const fileBaseName = fileName.replace('.pdf', '');  // 확장자 제거
    router.push(`/summary?file=${fileBaseName}`);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-center mb-8">Paper Summarizer</h1>
      <div className="grid md:grid-cols-2 gap-8">
        <Card>
          <CardHeader>
            <CardTitle>Upload New Paper</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="flex items-center space-x-4">
                <Input
                  type="file"
                  accept=".pdf"
                  onChange={handleFileChange}
                  className="flex-grow"
                />
                <Button type="submit" disabled={!file || isLoading}>
                  <UploadIcon className="mr-2 h-4 w-4" /> {isLoading ? 'Uploading...' : 'Upload'}
                </Button>
              </div>
              {file && <p className="text-sm text-muted-foreground">Selected file: {file.name}</p>}
            </form>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Previous Uploads</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {previousUploads.length > 0 ? (
                previousUploads.map((upload) => (
                  <li key={upload.id} className="flex items-center space-x-2 cursor-pointer" onClick={() => handlePreviousUploadClick(upload.name)}>
                    <FileTextIcon className="h-4 w-4 text-muted-foreground" />
                    <span>{upload.name}</span>
                  </li>
                ))
              ) : (
                <p className="text-sm text-muted-foreground">No previous uploads found.</p>
              )}
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* 진행 상태 표시 */}
      {isLoading && (
        <div style={{ width: '100%', backgroundColor: '#E5E7EB', borderRadius: '9999px', marginTop: '16px', height: '24px', position: 'relative', overflow: 'hidden' }}>
          {/* 로딩 바 */}
          <div
            style={{
              backgroundImage: 'linear-gradient(to right, #ACFFCC, #8CB0FF)',
              borderRadius: '9999px',
              transition: 'width 0.5s ease-in-out',
              width: `${progress}%`,
              height: '100%',
              position: 'absolute',
              top: 0,
              left: 0,
            }}
          ></div>

          {/* 로딩 바의 광택 효과 */}
          <div
            style={{
              position: 'absolute',
              inset: 0,
              backgroundImage: 'linear-gradient(to right, transparent, rgba(255, 255, 255, 0.5), transparent)',
              opacity: '0.5',
              borderRadius: '9999px',
              width: '50%',
              height: '100%',
              animation: 'pulse 1.5s infinite',
            }}
          ></div>

          {/* 퍼센트 텍스트 */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: `${3 + (progress * 0.94)}%`,  // 텍스트 위치를 progress 값에 맞게 조정
              transform: 'translateX(-50%)',  // 텍스트가 로딩 바 중앙에 위치하도록 조정
              height: '100%',
              width: '80%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '12px',
              fontWeight: 'bold',
              color: '#1F2937',
              zIndex: 1,
              whiteSpace: 'nowrap',  // 텍스트가 잘리지 않도록 설정
            }}
          >
            {progress}%
          </div>
        </div>
      )}
    </div>
  );
}