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

</style>
""", unsafe_allow_html=True)

st.title("🔰 Praçame - Suporte Técnico Militar")
st.write("Versão de testes — respondo dúvidas sobre **hardware**.")

# -------------------------
#  SESSION STATE
# -------------------------
if "historico" not in st.session_state:
    st.session_state["historico"] = []

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
st.markdown('<div class="chat-container" id="chatbox">', unsafe_allow_html=True)

# Renderizar histórico
for troca in st.session_state["historico"]:
    if troca["user"]:
        st.markdown(
            f"""
            <div class="msg-row user">
                <div class="user-msg">{troca["user"]}</div>
                <img class="avatar" src="https://i.imgur.com/TrVh7U1.png">
            </div>
            """,
            unsafe_allow_html=True,
        )

    if troca["bot"]:
        st.markdown(
            f"""
            <div class="msg-row">
                <img class="avatar" src="https://i.imgur.com/8cLZQvB.png">
                <div class="bot-msg">{troca["bot"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
#  INPUT DO CHAT
# -------------------------
pergunta = st.chat_input("Digite sua dúvida...")

if pergunta:

    # Adiciona no histórico
    st.session_state["historico"].append({"user": pergunta, "bot": None})

    # Vetor + busca
    vetor = embeddings.embed_query(pergunta)
    resultados = db.similarity_search_by_vector_with_relevance_scores(vetor, k=4)
    textos_resultado = [r[0].page_content for r in resultados]
    base_conhecimento = "\n----\n".join(textos_resultado)

    # Histórico formatado
    historico_formatado = ""
    for troca in st.session_state["historico"]:
        if troca["bot"]:
            historico_formatado += f"Usuário: {troca['user']}\nAssistente: {troca['bot']}\n"

    # Gera resposta com LangChain
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_injetado = prompt.invoke({
        "historico": historico_formatado,
        "base_conhecimento": base_conhecimento,
        "pergunta": pergunta
    })

    resposta_final = modelo.invoke(prompt_injetado).content

    # -------------------------
    #  EFEITO DE DIGITAÇÃO
    # -------------------------
    container_msg = st.empty()
    texto_parcial = ""

    for char in resposta_final:
        texto_parcial += char
        container_msg.markdown(
            f"""
            <div class="msg-row">
                <img class="avatar" src="https://i.imgur.com/8cLZQvB.png">
                <div class="bot-msg">{texto_parcial}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Salva texto final no histórico
    st.session_state["historico"][-1]["bot"] = resposta_final

    # --- Força scroll suave após a resposta ---
    st.markdown("""
    <script>
        const box = document.getElementById("chatbox");
        if (box) {
            box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' });
        }
    </script>
    """, unsafe_allow_html=True)


    # Recarrega interface
    st.rerun()

