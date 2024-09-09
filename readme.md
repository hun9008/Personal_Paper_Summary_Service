# Personal Paper Summary Assistent 

**FullStack Project**

| Part | Stack |
| --- | --- |
| FrontEnd | NextJS |
| BackEnd | FastAPI |
| Model | Llama3.1 VILA |

기존에 논문 summary 서비스들이 기능이 마음에 안들고 유료라 개인 사용을 위해 만들어 봄. Llama의 경우 로컬에서도 사용할 수 있어, 쓸만하지 않을까 하는 생각에 시작. 

## 2024-09-08

- 우선 python으로 pdf에서 텍스트 추출 후, Llama3.1로 summary 진행해 봤는데, summary가 괜찮음. 
  
- 그러나 논문 pdf는 좌우로 글이 나뉘어 있어, 좌우를 동시에 읽는 문제 존재.

- 찾아보니, pdfplumber를 사용하지 않고 VILA라는 이미지 모델로 논문을 section별로 추출한 깃허브 레포 발견. -> 적용해보니 잘됨.

- VILA로 추출한 section별 내용을 병렬로 Llama3.1에 질문할 예정이었으나, VILA에서 제공하는 라이브러리가 FastAPI의 asyncio와 fork 버전이 달라? (정확히는 어떤 에러인지 모르겠음) 병렬처리가 안되어 section별 순차적으로 요약 진행함.

- summary는 일관성을 위해 markdown으로 요약했고, section 별 중요한 부분은 빨간색으로 강조하도록 프롬프팅 진행.

- Naive Test 결과를 바탕으로 FastAPI 서버 구성하고, NextJS로 간단히 pdf업로드 하는 페이지와, pdf 원본 & 요약한 글을 보여주는 페이지 2개 제작.

- 백엔드에서 모델 서빙에 시간이 걸리기 때문에 프론트와 소켓을 통해 실시간 진행 상황을 받아오고 퍼센테이지 게이지로 표시.

- 프론트에서 markdown LaTex를 읽는데 문제가 있어, remarkMath 라이브러리 적용.

> 이후 강조된 부분이 중요한 이유를 설명하는 호버 이벤트를 추가할 예정
>
> 이후 호버한 부분을 포함해 Llama와 챗을 할 수 있는 기능 추가할 예정
>
> 현재는 이전 요약 기록을 로컬에 저장하고 로컬에 상태를 확인해 캐시된 요약을 가져오지만, DB로 Migration 할 예정.

