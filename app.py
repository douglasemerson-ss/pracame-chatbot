import streamlit as st
import time
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------------
# CONFIGURAÇÃO DO SISTEMA
# ------------------------------------------------------
st.set_page_config(page_title="Praçame Chatbot", page_icon="🔰", layout="wide")

st.markdown("""
<style>

body {
    background-color: #f2f2f2 !important;
}

.chat-wrapper {
    max-width: 900px;
    margin: auto;
}

.chat-container {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px;
    height: 70vh;
    overflow-y: auto;
    border: 1px solid #e5e5e5;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.msg-row {
    display: flex;
    margin-bottom: 14px;
}

.msg-row.user {
    justify-content: flex-end;
}

.bubble-user {
    background: #007aff;
    color: white;
    padding: 12px 16px;
    border-radius: 16px;
    max-width: 70%;
    font-size: 15px;
    line-height: 1.4;
}

.bubble-bot {
    background: #f1f0f0;
    color: #111;
    padding: 12px 16px;
    border-radius: 16px;
    max-width: 70%;
    font-size: 15px;
    line-height: 1.4;
}

.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    margin: 0 8px;
}

.typing {
    font-style: italic;
    color: #666;
}

</style>
""", unsafe_allow_html=True)

st.title("🔰 Praçame — Assistente Técnico Militar")
st.write("Chat moderno com histórico, scroll suave e UI profissional.")

# ------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------
if "historico" not in st.session_state:
    st.session_state["historico"] = []

if "digitando" not in st.session_state:
    st.session_state["digitando"] = False

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", None)
if not OPENAI_KEY:
    st.error("OPENAI_API_KEY não encontrada. Configure em Streamlit Secrets.")
    st.stop()

CAMINHO_DB = "db"

# ------------------------------------------------------
# MODEL / EMBEDDINGS
# ------------------------------------------------------
@st.cache_resource
def carregar_modelos():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model="text-embedding-3-small")
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embeddings)
    modelo = ChatOpenAI(openai_api_key=OPENAI_KEY, model="gpt-4o-mini", temperature=0.2)
    return embeddings, db, modelo

embeddings, db, modelo = carregar_modelos()

# ------------------------------------------------------
# PROMPT
# ------------------------------------------------------
prompt_template = """
INSTRUÇÕES IMPORTANTES:
- Responda SOMENTE usando a base de conhecimento abaixo.
- NÃO invente e NÃO use conhecimento fora da base.
- NÃO repita tags como "Usuário:" ou "Assistente:".

Base de conhecimento:
{base_conhecimento}

Histórico (resumo):
{historico}

Pergunta:
{pergunta}

Resposta:
"""

# ------------------------------------------------------
# CHAT RENDER
# ------------------------------------------------------
def render_chat():
    st.markdown('<div id="chatbox" class="chat-container">', unsafe_allow_html=True)

    for troca in st.session_state["historico"]:
        # Mensagem do usuário
        st.markdown(
            f"""
            <div class="msg-row user">
                <div class="bubble-user">{troca['user']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Resposta do bot
        if troca.get("bot"):
            st.markdown(
                f"""
                <div class="msg-row">
                    <div class="bubble-bot">{troca['bot']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Indicador "digitando..."
    if st.session_state["digitando"]:
        st.markdown(
            """
            <div class="msg-row">
                <div class="bubble-bot typing">Digitando...</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Scroll automático somente após nova mensagem
    st.markdown("""
        <script>
        const box = document.getElementById("chatbox");
        if (box) {
            setTimeout(() => { box.scrollTop = box.scrollHeight; }, 120);
        }
        </script>
    """, unsafe_allow_html=True)


# ------------------------------------------------------
# RENDER INICIAL
# ------------------------------------------------------
render_chat()

# ------------------------------------------------------
# INPUT DO CHAT
# ------------------------------------------------------
pergunta = st.chat_input("Digite sua dúvida...")

if pergunta:

    # Adiciona pergunta ao histórico
    st.session_state["historico"].append({"user": pergunta, "bot": None})

    # Ativa indicador digitando
    st.session_state["digitando"] = True
    render_chat()

    # PROCESSAMENTO
    vetor = embeddings.embed_query(pergunta)
    resultados = db.similarity_search_by_vector_with_relevance_scores(vetor, k=6)

    if not resultados:
        resposta_final = "Não encontrei informações suficientes na base de conhecimento."
    else:
        textos = [doc.page_content for doc, score in resultados]
        base_conhecimento = "\n\n----\n\n".join(textos)

        resumo = []
        for troca in st.session_state["historico"][:-1]:
            if troca.get("bot"):
                resumo.append(f"User: {troca['user']}\nAssistant: {troca['bot']}")
            else:
                resumo.append(f"User: {troca['user']}")

        resumo_txt = "\n\n".join(resumo[-6:])

        prompt = ChatPromptTemplate.from_template(prompt_template)
        prompt_injetado = prompt.invoke({
            "base_conhecimento": base_conhecimento,
            "historico": resumo_txt,
            "pergunta": pergunta
        })

        resposta_final = modelo.invoke(prompt_injetado).content

        if not resposta_final or len(resposta_final.strip()) < 5:
            resposta_final = "Não encontrei informações suficientes na base de conhecimento."

    # Salva resposta
    st.session_state["historico"][-1]["bot"] = resposta_final

    # Desliga digitando
    st.session_state["digitando"] = False
    render_chat()
