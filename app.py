import streamlit as st
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

CAMINHO_DB = "db"

prompt_template = """
Você é um assistente técnico militar especializado em suporte ao usuário.
Todos os usuários são leigos no assunto, imagine que são crianças lidando com problemas de T.I.

Histórico da conversa:
{historico}

Base de conhecimento relevante:
{base_conhecimento}

Pergunta atual do usuário:
{pergunta}

Explique a causa do problema e ofereça soluções de forma super didática, calma,
clara e com um linguajar simples.
"""

# ---- CONFIGURAÇÃO STREAMLIT ----
st.set_page_config(page_title="Praçame Chatbot", page_icon="🔰")
st.header("🔰 Praçame - Suporte Técnico Militar")
st.write("Estou em versão de testes, apenas respondo algumas perguntas sobre Hardware")

# Inicializar sessão
if "historico" not in st.session_state:
    st.session_state["historico"] = []

# carregar modelo e base
@st.cache_resource
def carregar_modelos():
    api_key = st.secrets["OPENAI_API_KEY"]

    embeddings = OpenAIEmbeddings(api_key=api_key)
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embeddings)

    modelo = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",   # seguro + barato + rápido
        temperature=0.4
    )

    return embeddings, db, modelo

embeddings, db, modelo = carregar_modelos()

# Campo de input
pergunta = st.chat_input("Digite sua dúvida...")

if pergunta:
    # adicionar pergunta ao chat
    st.session_state["historico"].append({"user": pergunta, "bot": None})

    # buscar informações relevantes
    vetor = embeddings.embed_query(pergunta)
    resultados = db.similarity_search_by_vector_with_relevance_scores(vetor, k=4)
    textos_resultado = [r[0].page_content for r in resultados]
    base_conhecimento = "\n\n----\n\n".join(textos_resultado)

    # montar histórico
    historico_formatado = ""
    for troca in st.session_state["historico"]:
        if troca["bot"] is not None:
            historico_formatado += f"Usuário: {troca['user']}\nAssistente: {troca['bot']}\n"

    # gerar resposta
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_injetado = prompt.invoke({
        "historico": historico_formatado,
        "base_conhecimento": base_conhecimento,
        "pergunta": pergunta
    })

    resposta = modelo.invoke(prompt_injetado).content

    # salvar e exibir
    st.session_state["historico"][-1]["bot"] = resposta

# mostrar histórico no chat
for troca in st.session_state["historico"]:
    with st.chat_message("user"):
        st.write(troca["user"])
    if troca["bot"]:
        with st.chat_message("assistant"):
            st.write(troca["bot"])
