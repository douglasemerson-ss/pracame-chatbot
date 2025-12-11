import streamlit as st
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# =========================
# Configuração da página
# =========================
st.set_page_config(page_title="Praçame Chatbot", page_icon="Flag", layout="centered")

st.title("Flag Praçame - Suporte Técnico Militar")
st.markdown("Este chatbot foi desenvolvido para reduzir a abertura de chamados de TI.")
st.markdown("**Versão de testes** — respondo apenas sobre o **Assinador SERPRO**.")
st.markdown("---")

# =========================
# CSS bonito e funcional
# =========================
st.markdown("""
<style>
    .chat-container {
        padding: 0;
        margin: 0;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        height: 65vh;
        overflow-y: auto;
        padding: 20px 10px 100px 10px;
        scroll-behavior: smooth;
    }
    .user-bubble {
        background: #d1e7ff;
        color: black;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        margin: 8px 0 8px auto;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .bot-bubble {
        background: #e8f5e8;
        color: black;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        margin: 8px auto 8px 0;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .typing {
        background: #e8f5e8;
        color: #555;
        font-style: italic;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        max-width: 100px;
        margin: 8px auto 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Session state
# =========================
if "historico" not in st.session_state:
    st.session_state.historico = []
if "digitando" not in st.session_state:
    st.session_state.digitando = False

# =========================
# Chave da OpenAI
# =========================
if not st.secrets.get("OPENAI_API_KEY"):
    st.error("Configure a OPENAI_API_KEY nos Secrets do Streamlit")
    st.stop()

# =========================
# Carrega modelos (cache)
# =========================
@st.cache_resource
def load_models():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    return embeddings, db, llm

embeddings, db, llm = load_models()

# =========================
# Prompt rígido anti-alucinação
# =========================
prompt_template = """
Você é um assistente de suporte técnico militar extremamente preciso.
Responda APENAS com base nos documentos abaixo. Nunca invente informação.

Documentos relevantes:
{context}

Histórico da conversa (para contexto apenas):
{history}

Pergunta do usuário:
{question}

Resposta clara e didática (passo a passo quando necessário):
"""

# =========================
# Função que renderiza o chat uma única vez
# =========================
chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-container" id="chatbox">', unsafe_allow_html=True)

    for msg in st.session_state.historico:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

    # Indicador de digitação
    if st.session_state.digitando:
        st.markdown('<div class="typing">Digitando...</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Input do usuário
# =========================
if pergunta := st.chat_input("Digite sua dúvida sobre o Assinador SERPRO..."):
    # 1. Adiciona pergunta do usuário
    st.session_state.historico.append({"role": "user", "content": pergunta})
    st.session_state.digitando = True

    # Re-renderiza imediatamente (mostra a pergunta + "Digitando...")
    st.rerun()

# =========================
# Geração da resposta (só roda quando tem pergunta nova e ainda não respondeu)
# =========================
if st.session_state.digitando:
    ultima_pergunta = st.session_state.historico[-1]["content"]

    # Busca no vetor
    docs = db.similarity_search_with_relevance_scores(ultima_pergunta, k=6)
    
    if not docs or docs[0][1] < 0.7:  # score muito baixo = sem informação confiável
        resposta = "Desculpe, não encontrei informações suficientes na base de conhecimento para responder essa dúvida com segurança."
    else:
        context = "\n\n---\n\n".join([doc.page_content])
        #context = "\n\n---\n\n".join([doc.page_content for doc, _ in docs])
        # Monta histórico limpo (últimas 6 trocas)
        history = ""
        for m in st.session_state.historico[:-1][-6:]:
            role = "Usuário" if m["role"] == "user" else "Assistente"
            history += f"{role}: {m['content']}\n"
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        
        resposta = chain.invoke({
            "context": context,
            "history": history,
            "question": ultima_pergunta
        }).content

    # Salva resposta
    st.session_state.historico.append({"role": "assistant", "content": resposta})
    st.session_state.digitando = False

    # Força atualização da tela
    st.rerun()

# =========================
# Scroll automático só quando tem conversa
# =========================
if len(st.session_state.historico) > 0 or st.session_state.digitando:
    st.markdown("""
    <script>
        const chatbox = document.getElementById("chatbox");
        if (chatbox) {
            chatbox.scrollTop = chatbox.scrollHeight;
        }
    </script>
    """, unsafe_allow_html=True)
