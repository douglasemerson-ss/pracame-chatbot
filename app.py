from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import openai
from dotenv import load_dotenv

# Carregar secret key
openai.api_key = st.secrets["OPENAI_API_KEY"]



load_dotenv()

CAMINHO_DB = "db"

prompt_template = """
Você é um assistente técnico militar especializado em suporte ao usuário. 
Todos os usuários são leigos no assunto, imagine que são crianças lidando com problemas de T.I

Histórico da conversa:
{historico}

Base de conhecimento relevante:
{base_conhecimento}

Pergunta atual do usuário:
{pergunta}

Você precisa explicar a razão do problema e dar as soluções de forma super didáditica e clara com um linguajar simples
"""

def iniciar_chatbot():

    print("🔰 Praçame iniciado! Digite 'sair' para encerrar.\n")

    historico_conversa = []
    funcao_embedding = OpenAIEmbeddings(api_key=openai.api_key)
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)
    modelo = ChatOpenAI(api_key=openai.api_key)
    
    while True:
        pergunta = input("Você: ")

        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("Encerrando o chat. Até mais!")
            break

        # converter pergunta em embedding e buscar no DB
        vetor = funcao_embedding.embed_query(pergunta)
        resultados = db.similarity_search_by_vector_with_relevance_scores(vetor, k=4)

        textos_resultado = [r[0].page_content for r in resultados]
        base_conhecimento = "\n\n----\n\n".join(textos_resultado)

        # montar histórico concatenado
        historico_formatado = ""
        for troca in historico_conversa:
            historico_formatado += f"Usuário: {troca['user']}\nAssistente: {troca['bot']}\n"

        # gerar resposta
        prompt = ChatPromptTemplate.from_template(prompt_template)
        prompt_injetado = prompt.invoke({
            "historico": historico_formatado,
            "base_conhecimento": base_conhecimento,
            "pergunta": pergunta
        })

        resposta = modelo.invoke(prompt_injetado).content
        print("\nPraçame:", resposta, "\n")

        # salvar no histórico
        historico_conversa.append({"user": pergunta, "bot": resposta})

# iniciar chatbot
iniciar_chatbot()
