import streamlit as st
import time
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="Praçame Chatbot", page_icon="Flag", layout="wide")

st.title("Flag Praçame - Suporte Técnico Militar")
st.header("Este chatbot foi desenvolvido a partir da necessidade da equipe de TI para diminuir o fluxo de abertura de chamados.")
st.subheader("Atualmente sou uma versão de testes — respondo dúvidas sobre **Assinador SERPRO**.")
st.markdown("---")

# -------------------------
# CSS / estilos
# -------------------------
st.markdown("""
<style>
.chat-container {
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
    padding-bottom: 100px;
    height: 70vh;
    overflow-y: auto;
    scroll-behavior: smooth;
    border: 1px solid #ddd;
    border-radius: 12px;
    background: #fafafa;
}
.user-msg {
    background: #d9e6ff;
    color: #000;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 8px 12px 8px 80px;
    max-width: 75%;
    align-self: flex-end;
    word-wrap: break-word;
}
.bot-msg {
    background: #eef5e8;
    color: #000;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 8px 80px 8px 12px;
    max-width: 75%;
    align-self: flex-start;
    word-wrap: break-word;
}
.msg-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 10px;
}
.msg-row.user {
    justify-content: flex-end;
}
.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    margin: 0 10px;
    flex-shrink: 0;
}
.typing {
    font-style: italic;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session state
# -------------------------
if "historico" not in st.session_state:
    st.session_state["historico"] = []           # [{"user": "...", "bot": "..."}]
if "digitando" not in st.session_state:
    st.session_state["digitando"] = False

# -------------------------
# Load OpenAI key
# -------------------------
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", None)
if not OPENAI_KEY:
    st.error("OPENAI_API_KEY não encontrada em Streamlit Secrets. Vá em Manage App → Secrets e adicione.")
    st.stop()

CAMINHO_DB = "db"

# -------------------------
# Prompt seguro
# -------------------------
prompt_template = """
INSTRUÇÕES IMPORTANTES (LIMITE RÍGIDO):
- Você só pode responder usando EXCLUSIVAMENTE a "Base de conhecimento fornecida abaixo.
- NÃO invente, NÃO adivinhe e NÃO use conhecimento externo.
- NÃO repita literalmente as marcações do histórico (ex: "Usuário:", "Assistente:") na sua resposta.
- Seja didático, use linguagem simples e explique causas e passos de solução.

Base de conhecimento (trechos recuperados):
{base_conhecimento}

Histórico resumido (apenas para contexto, NÃO repita marcações):
{historico}

Pergunta atual:
{pergunta}

Resposta:
"""

# -------------------------
# Carregar modelos e DB (cache)
# -------------------------
@st.cache_resource
def carregar_modelos():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model="text-embedding-3-small")
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embeddings)
    modelo = ChatOpenAI(openai_api_key=OPENAI_KEY, model="gpt-4o-mini", temperature=0.2)
    return embeddings, db, modelo

embeddings, db, modelo = carregar_modelos()

# -------------------------
# Container do chat
# -------------------------
chat_box = st.container()

def render_chat():
    with chat_box:
        st.markdown('<div id="chatbox" class="chat-container">', unsafe_allow_html=True)
        
        for troca in st.session_state["historico"]:
            # Mensagem do usuário
            st.markdown(
                f"""
                <div class="msg-row user">
                    <div class="user-msg">{troca['user']}</div>
                    <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/9977/9977334.png">
                </div>
                """,
                unsafe_allow_html=True
            )
            # Mensagem do bot
            if troca.get("bot"):
                st.markdown(
                    f"""
                    <div class="msg-row">
                        <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/7985/7985432.png">
                        <div class="bot-msg">{troca['bot']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Indicador digitando
        if st.session_state["digitando"]:
            st.markdown(
                f"""
                <div class="msg-row">
                    <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/7985/7985432.png">
                    <div class="bot-msg typing">Digitando...</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

# Renderização inicial
render_chat()

# -------------------------
# Input do usuário
# -------------------------
pergunta = st.chat_input("Digite sua dúvida sobre o Assinador SERPRO...")

if pergunta:
    # 1. Adiciona mensagem do usuário
    st.session_state["historico"].append({"user": pergunta, "bot": None})
    
    # 2. Mostra "digitando..."
    st.session_state["digitando"] = True
    render_chat()

    # 3. Gera resposta
    vetor = embeddings.embed_query(pergunta)
    resultados = db.similarity_search_by_vector_with_relevance_scores(vetor, k=6)

    if not resultados:
        resposta_final = "Não encontrei informações suficientes na base de conhecimento para responder a isso."
    else:
        base_conhecimento = "\n\n----\n\n".join([doc.page_content for doc, _ in resultados])
        
        # Histórico resumido (últimas 6 trocas
        resumo = []
        for t in st.session_state["historico"][:-1][-6:]:
            if t.get("bot"):
                resumo.append(f"User: {t['user']}\nAssistant: {t['bot']}")
            else:
                resumo.append(f"User: {t['user']}")
        historico_texto = "\n\n".join(resumo)

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | modelo
        resposta_final = chain.invoke({
            "base_conhecimento": base_conhecimento,
            "historico": historico_texto,
            "pergunta": pergunta
        }).content

        # Segurança extra
        if not resposta_final or len(resposta_final.strip()) < 8:
            resposta_final = "Desculpe, não consegui encontrar uma resposta clara na base de conhecimento."

    # 4. Salva resposta
    st.session_state["historico"][-1]["bot"] = resposta_final
    st.session_state["digitando"] = False

    # 5. Re-renderiza com a resposta
    render_chat()

# SCROLL INTELIGENTE — só rola para baixo se já houver conversa
if st.session_state["historico"] or st.session_state["digitando"]:
    st.markdown("""
    <script>
        const box = document.getElementById("chatbox");
        if (box) {
            box.scrollTop = box.scrollHeight;
        }
    </script>
    """, unsafe_allow_html=True)
