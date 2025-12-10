import streamlit as st
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time

# -------------------------
#  CONFIG STREAMLIT
# -------------------------
st.set_page_config(
    page_title="Praçame Chatbot",
    page_icon="🔰",
    layout="wide"
)

# --- CSS personalizado ---
st.markdown("""
<style>
.chat-container {
    max-width: 850px;
    margin-left: auto;
    margin-right: auto;
    padding-bottom: 90px;
}

.user-msg {
    background: #d9e6ff;
    color: #000;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 6px 0;
    width: fit-content;
    max-width: 80%;
}

.bot-msg {
    background: #eef5e8;
    color: #000;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 6px 0;
    width: fit-content;
    max-width: 80%;
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
    width: 36px;
    height: 36px;
    border-radius: 50%;
    margin: 0 8px;
}

.scroll-fix {
    height: 30px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔰 Praçame - Suporte Técnico Militar")
st.write("Versão de testes — respondo dúvidas sobre **hardware**.")

# -------------------------
#  SESSION STATE
# -------------------------
if "historico" not in st.session_state:
    st.session_state["historico"] = []

if "digitando" not in st.session_state:
    st.session_state["digitando"] = False

OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
CAMINHO_DB = "db"

prompt_template = """
Você é um assistente técnico militar especializado em suporte ao usuário.
Explique sempre de forma didática, calma e clara, usando linguajar simples.

Histórico da conversa:
{historico}

Base de conhecimento relevante:
{base_conhecimento}

Pergunta atual:
{pergunta}
"""

@st.cache_resource
def carregar_modelos():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embeddings)
    modelo = ChatOpenAI(openai_api_key=OPENAI_KEY)
    return embeddings, db, modelo

embeddings, db, modelo = carregar_modelos()

# -------------------------
#  ÁREA DE CHAT
# -------------------------
container_chat = st.container()
scroll_anchor = st.empty()

with container_chat:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for troca in st.session_state["historico"]:
        if troca["user"]:
            st.markdown(
                f"""
                <div class="msg-row user">
                    <div class="user-msg">{troca["user"]}</div>
                    <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/9977/9977334.png">
                </div>
                """,
                unsafe_allow_html=True
            )

        if troca["bot"]:
            st.markdown(
                f"""
                <div class="msg-row">
                    <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/7985/7985432.png">
                    <div class="bot-msg">{troca["bot"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # placeholder para o efeito "digitando..."
    if st.session_state["digitando"]:
        st.markdown(
            """
            <div class="msg-row">
                <img class="avatar" src="https://i.imgur.com/8cLZQvB.png">
                <div class="bot-msg"><i>Digitando...</i></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

scroll_anchor.markdown('<div class="scroll-fix"></div>', unsafe_allow_html=True)

# -------------------------
#  INPUT
# -------------------------
pergunta = st.chat_input("Digite sua dúvida...")

if pergunta:

    # A mensagem do usuário aparece no chat imediatamente
    st.session_state["historico"].append({"user": pergunta, "bot": None})

    # Ativa indicador de digitação
    st.session_state["digitando"] = True
    st.rerun()

# Se está "digitando", gerar resposta agora
if st.session_state["digitando"]:

    ultima_msg = st.session_state["historico"][-1]["user"]

    vetor = embeddings.embed_query(ultima_msg)
    resultados = db.similarity_search_by_vector_with_relevance_scores(vetor, k=4)
    textos_resultado = [r[0].page_content for r in resultados]
    base_conhecimento = "\n----\n".join(textos_resultado)

    historico_formatado = ""
    for troca in st.session_state["historico"][:-1]:  # evita mensagem sem bot
        if troca["bot"]:
            historico_formatado += f"Usuário: {troca['user']}\nAssistente: {troca['bot']}\n"

    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_injetado = prompt.invoke({
        "historico": historico_formatado,
        "base_conhecimento": base_conhecimento,
        "pergunta": ultima_msg
    })

    resposta_final = modelo.invoke(prompt_injetado).content

    # salva no histórico
    st.session_state["historico"][-1]["bot"] = resposta_final

    # desativa "digitando"
    st.session_state["digitando"] = False

    # rerun para exibir bot e rolar
    time.sleep(0.1)
    st.rerun()
