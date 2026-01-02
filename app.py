# app.py
import streamlit as st
import os
import pandas as pd
import torch
import gc
import re
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# === 페이지 설정 ===
st.set_page_config(page_title="입찰메이트 AI", page_icon="🤖", layout="wide")
st.title("🤖 입찰메이트: 공공사업 입찰 AI 컨설턴트")

# === 설정 및 경로 ===
MODEL_ID = "MyeongHo0621/Qwen2.5-3B-Korean"
BASE_DIR = "."
DB_PATH = os.path.join(BASE_DIR, "data", "vector_store", "chroma_db")
CSV_PATH = os.path.join(BASE_DIR, "data", "data_list.csv")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "data", "model_cache")

# === 1. 리소스 로드 (캐싱 적용: 새로고침해도 다시 로드 안 함) ===
@st.cache_resource
def load_resources():
    print("🚀 리소스 로딩 시작...")
    
    # (1) 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>"), tokenizer.convert_tokens_to_ids("<|im_end|>")]
    
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=512, temperature=0.1, repetition_penalty=1.2,
        return_full_text=False, eos_token_id=terminators, do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # (2) 임베딩 & DB 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="dragonkue/BGE-m3-ko",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder=MODEL_CACHE_DIR
    )
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings, collection_name="government_proposals")
    
    # (3) CSV 힌트 맵 로드
    cheat_sheet = {}
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            filename = str(row.get('파일명', ''))
            budget = row.get('사업 금액', 0)
            if not filename or filename == 'nan': continue
            file_stem = os.path.splitext(filename)[0]
            try: formatted = f"{int(budget):,}원"
            except: formatted = str(budget)
            cheat_sheet[file_stem] = formatted
            
    print("✅ 리소스 로딩 완료!")
    return llm, vector_store, cheat_sheet

# 로딩 실행 (스피너 표시)
with st.spinner("AI 컨설턴트가 출근 준비 중입니다... (모델 로드 중)"):
    llm, vector_store, budget_map = load_resources()

# === 2. 유틸리티 함수들 (검색, 검증, 후처리) ===
def retrieve_docs(query, k=3):
    # 간단하게 키워드 없이 전체 검색 (실제 사용성을 위해)
    # 필요하면 사용자에게 타겟 키워드를 입력받게 UI 수정 가능
    return vector_store.similarity_search(query, k=k)

def verify_facts(doc_content, query):
    risk_keywords = ["블록체인", "AI", "인공지능", "메타버스", "NFT", "클라우드", "빅데이터"]
    # 질문에 포함된 위험 키워드 추출
    target_kws = [kw for kw in risk_keywords if kw.lower() in query.lower()]
    
    if not target_kws:
        return True, "" # 안전함 (검증할 키워드 없음)
    
    # 문서 내용에 키워드가 있는지 확인
    missing = [kw for kw in target_kws if doc_content.lower().count(kw.lower()) == 0]
    
    if missing:
        # 하나라도 없으면 False 리턴
        msg = f"문서 검토 결과, 질문하신 기술 키워드 '{', '.join(missing)}'은(는) 제안요청서에 포함되어 있지 않습니다."
        return False, msg
        
    return True, "키워드 확인됨" # 키워드가 문서에 실제로 있음

def finalize_text(text):
    text = text.replace("won", "원").replace("WON", "원").replace("<|im_end|>", "")
    return text.strip()

# === 3. 채팅 UI 구성 ===

# 세션 상태 초기화 (대화 기록 저장)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("사업에 대해 궁금한 점을 물어보세요 (예: 벤처확인 사업 예산은?)"):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 답변 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # (1) 검색
        docs = retrieve_docs(prompt, k=3) 
        
        if not docs:
            full_response = "죄송합니다. 관련 문서를 찾지 못했습니다."
            message_placeholder.markdown(full_response)
        else:
            best_doc = docs[0]
            filename = os.path.basename(best_doc.metadata['source'])
            file_stem = os.path.splitext(filename)[0]
            
            # (2) 팩트 검증 (가드레일) ★ 여기가 핵심 변경점 ★
            is_safe, fact_msg = verify_facts(best_doc.page_content, prompt)
            
            # (3) 힌트 준비 (예산)
            budget_hint = "정보 없음"
            for key, val in budget_map.items():
                if key in file_stem or file_stem in key:
                    budget_hint = val; break

            # === [Guardrail Logic] ===
            # 검증 결과가 '안전하지 않음(False)'이면 LLM을 부르지 않고 바로 차단!
            if not is_safe:
                # LLM 생성 생략 -> Python이 직접 답변
                full_response = f"🚫 **팩트 체크 경고**\n\n{fact_msg}\n\n(AI 환각 방지를 위해 답변 생성을 중단했습니다.)"
                full_response += f"\n\n--- \n📄 **참고 문서:** `{filename}`"
                message_placeholder.markdown(full_response)
                
            else:
                # 안전한 경우에만 LLM 호출 (기존 로직)
                if "예산" in prompt or "금액" in prompt:
                    template = """<|im_start|>system
당신은 '숫자 확인 봇'입니다.
오직 [DB 힌트]에 있는 금액을 확인하여 정답을 말하세요.
[절대 규칙]
1. 예산 금액은 반드시 '원' 단위로 답하세요. (표기 금액은 달러($)가 아닌 원화입니다.)
2. [힌트]에 있는 금액은 $가 아닌 원화(￦) 단위입니다. 숫자를 임의로 바꾸거나 환율 계산을 금지합니다.
3. 숫자는 '350,000,000원' 처럼 한화 단위로 정확히 표기하세요. '$'는 쓰지 않습니다.
<|im_end|>
<|im_start|>user
[DB 힌트]
{hint}

[문서 내용]
{context}

[질문]
{question}
<|im_end|>
<|im_start|>assistant
"""
                    final_hint = budget_hint
                elif "요약" in prompt:
                    template = """<|im_start|>system
당신은 유능한 요약 전문가입니다.
문서의 핵심 내용을 파악하여 요청사항에 맞게 요약하세요.
[절대 규칙]
1. 한국어로 명확하게 작성하세요. (영어 사용은 피하세요.)
2. 가장 중요한 내용을 추려서 번호 매기기(1., 2., 3.)로 나열하세요.
3. 3가지일 경우 작성이 끝나면 즉시 답변을 멈추세요. (반복하지 마세요)
<|im_end|>
<|im_start|>user
[문서 내용]
{context}

[질문]
{question}
<|im_end|>
<|im_start|>assistant
핵심 요약:
"""
                    final_hint = ""
                else:
                    template = """<|im_start|>system
당신은 정직한 AI 컨설턴트입니다.
[검증 결과]를 절대적으로 신뢰하세요.
만약 [검증 결과]에서 "키워드가 없다"고 하면, 절대 거짓말하지 말고 "관련 내용 없음"이라고 답하세요.
<|im_end|>
<|im_start|>user
[검증 결과]
{hint}

[문서 내용]
{context}

[질문]
{question}
<|im_end|>
<|im_start|>assistant
"""
                    final_hint = ""

                # 생성
                prompt_obj = PromptTemplate.from_template(template)
                chain = prompt_obj | llm | StrOutputParser()
                
                try:
                    with st.spinner("답변 생성 중..."):
                        raw_response = chain.invoke({
                            "context": best_doc.page_content,
                            "question": prompt,
                            "hint": final_hint
                        })
                    
                    full_response = finalize_text(raw_response)
                    full_response += f"\n\n--- \n📄 **참고 문서:** `{filename}`"
                    message_placeholder.markdown(full_response)
                    
                except Exception as e:
                    message_placeholder.error(f"오류 발생: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})