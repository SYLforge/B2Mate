# 🤖 BidMate: 입찰메이트 (B2G RAG Consultant)

**Hybrid RAG & Fact Verification System for Government RFPs**

> **📊 프로젝트 결과 보고서**
> - 📄 **PDF 결과 보고서:** https://docs.google.com/document/d/19myvAj9gQ_1bXFT55bMx1jBiDv_rUcoj3lZYOb5hk1U/edit?usp=sharing
> - 📢 **발표용 보고서:** https://docs.google.com/presentation/d/19pgIaUCeuq35bHzv2fJfzbXE43ztQ06B

---

## 📌 프로젝트 개요
'입찰메이트(BidMate)'는 공공기관 제안요청서(RFP)를 분석하여 컨설턴트에게 핵심 정보(예산, 일정, 과업 내용)를 제공하는 **AI 입찰 컨설팅 시스템**입니다.
단순한 RAG(Retrieval-Augmented Generation)의 한계인 숫자(예산) 오류와 환각(Hallucination) 현상을 극복하기 위해, **정형 데이터(CSV)와 비정형 데이터(문서)를 결합한 Hybrid 아키텍처**를 구현했습니다.

## 🚀 주요 기능 (Key Features)

### 1. Hybrid RAG Architecture 
- **Vector Search:** `ChromaDB`와 `BGE-m3` 임베딩을 사용하여 문서 내 맥락 검색.
- **Metadata Injection:** `data_list.csv`의 정형 데이터를 힌트로 주입하여 **예산 정보 정확도 100%** 달성.

### 2. Hallucination Guardrail (Fact Verifier)
- **Python 기반 팩트 검증:** LLM이 생성하기 전, Python 코드가 문서 내 핵심 키워드(AI, 블록체인 등) 존재 여부를 사전 검증.
- **Hard Block:** 문서에 없는 내용을 질문할 경우, LLM 생성을 차단하고 즉시 "관련 내용 없음"을 출력.

### 3. Prompt Routing & Optimization
- **Mode Switching:** 질문 의도(예산/요약/일반)에 따라 최적화된 프롬프트 자동 적용.
- **Post-processing:** LLM의 영어 혼용(won, dollar) 문제를 Python 후처리로 해결하여 자연스러운 한국어 출력.

## 🛠️ 기술 스택 (Tech Stack)
- **Model:** `Qwen2.5-3B-Korean` (bfloat16)
- **Framework:** LangChain, Streamlit
- **Vector DB:** ChromaDB
- **Embedding:** dragonkue/BGE-m3-ko
- **Tools:** PyMuPDF, Pandas

## 📂 설치 및 실행 방법

### 환경 설정
```bash
# Repository 클론
git clone [https://github.com/SYLforge/B2Mate.git](https://github.com/SYLforge/B2Mate.git)
cd BidMate-RAG

# 패키지 설치
pip install -r requirements.txt
```

### 📂 디렉토리 구조 (Directory Structure)
실제 데이터 파일(`data_list.csv` 및 문서 파일)은 저장소에 포함되어 있지 않습니다.
프로젝트 실행을 위해서는 아래 구조에 맞춰 데이터를 `data/` 폴더에 위치시켜야 합니다.

```text
BidMate-RAG/
├── app.py                  # Streamlit 웹 애플리케이션 메인
├── 01_data_preprocessing.ipynb
├── 02_embedding.ipynb
├── 03_generation.ipynb
├── requirements.txt        # 필요 라이브러리 목록
├── README.md
└── data/                   # [데이터 폴더]
    ├── data_list.csv       # (필수) 메타데이터 파일 (사업명, 예산 등 포함)
    └── processed/          # (선택) 전처리된 텍스트 문서들
```

### 실행 순서 및 파일 설명

01_data_preprocessing.ipynb: 원본 문서를 텍스트로 변환 및 정제.

02_embedding.ipynb: 텍스트를 임베딩하여 chroma_db 폴더 생성.

03_generation.ipynb: RAG 로직 및 검증기 테스트
