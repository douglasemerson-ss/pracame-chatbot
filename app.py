import streamlit as st
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# -------------------------
#  CONFIGURAÇÕES STREAMLIT
# -------------------------
st.set_page_config(page_title="Praçame Chatbot", page_icon="🔰")
st.header("🔰 Praçame - Suporte Técnico Militar")
st.write("Estou em versão de testes, respondo dúvidas sobre problemas de hardware.")

# -------------------------
#  CARREGAR OPENAI API KEY
# -------------------------
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]

CAMINHO_DB = "db"

prompt_template = """
Você é um assistente técnico militar especializado em suporte ao usuário.
Todos os usuários são leigos no assunto, imagine que são crianças lidando com problemas de T.I.

Histórico da conversa até agora:
{historico}

Base de conhecimento relevante da documentação:
{base_conhecimento}

Pergunta atual do usuário:
{pergunta}

Explique a causa do problema e ofereça soluções de forma super didática, calma,
clara e com um linguajar simples.
"""

# -------------------------
#  ESTADO DA SESSÃO
# -------------------------
if "historico" not in st.session_state:
    st.session_state["historico"] = []

# -------------------------
#  CARREGAR MODELO + DB
# -------------------------
@st.cache_resource
def carregar_modelos():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embeddings)
    modelo = ChatOpenAI(openai_api_key=OPENAI_KEY)
    return embeddings, db, modelo

embeddings, db, modelo = carregar_modelos()

# -------------------------
#  CAMPO DE INPUT
# -------------------------
pergunta = st.chat_input("Digite sua dúvida...")

if pergunta:

    st.session_state["historico"].append({"user": pergunta, "bot": None})

    # ---- BUSCAR NO BANCO DE VETORES ----
    vetor = embeddings.embed_query(pergunta)
    resultados = db.similarity_search_by_vector_with_relevance_scores(vetor, k=4)

    textos_resultado = [r[0].page_content for r in resultados]
    base_conhecimento = "\n\n----\n\n".join(textos_resultado)

    # ---- HISTÓRICO FORMATADO ----
    historico_formatado = ""
    for troca in st.session_state["historico"]:
        if troca["bot"]:
            historico_formatado += f"Usuário: {troca['user']}\nAssistente: {troca['bot']}\n"

    # ---- GERAR RESPOSTA ----
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_injetado = prompt.invoke({
        "historico": historico_formatado,
        "base_conhecimento": base_conhecimento,
        "pergunta": pergunta
    })

    resposta = modelo.invoke(prompt_injetado).content

    # salvar no histórico
    st.session_state["historico"][-1]["bot"] = resposta

# -------------------------
#  MOSTRAR MENSAGENS
# -------------------------
for troca in st.session_state["historico"]:
    with st.chat_message("user"):
        st.write(troca["user"])
    if troca["bot"]:
        with st.chat_message("assistant"):
            st.write(troca["bot"])
