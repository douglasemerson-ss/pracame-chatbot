import streamlit as st
import time
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="Praçame Chatbot", page_icon="🔰", layout="wide")
st.title("🔰 Praçame - Suporte Técnico Militar")
st.header("Este chatbot foi desenvolvido a partir da necessidade da equipe de TI para diminuir o fluxo de abertura de chamados.")
st.subheader("Atualmente sou uma versão de testes — respondo dúvidas sobre **Assinador SERPRO**.")

# -------------------------
# CSS / estilos
# -------------------------
st.markdown("""
<style>
.chat-container {
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
    padding-bottom: 90px;
    height: 70vh;
    overflow-y: auto;
    scroll-behavior: smooth;
}
.user-msg {
    background: #d9e6ff;
    color: #000;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 6px 0;
    width: fit-content;
    max-width: 75%;
    word-wrap: break-word;
}
.bot-msg {
    background: #eef5e8;
    color: #000;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 6px 0;
    width: fit-content;
    max-width: 75%;
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
    width: 36px;
    height: 36px;
    border-radius: 50%;
    margin: 0 8px;
}
.typing {
    font-style: italic;
    color: #666;
}
 .scroll-fix { 
height: 10px; 
}
</style>
""", unsafe_allow_html=True)

# .scroll-fix { 
#height: 10px; 
#}


# -------------------------
# Session state
# -------------------------
if "historico" not in st.session_state:
    st.session_state["historico"] = []  # cada item: {"user": "...", "bot": "..."}

if "digitando" not in st.session_state:
    st.session_state["digitando"] = False

# -------------------------
# Load OpenAI key from Streamlit secrets
# -------------------------
# (Coloque OPENAI_API_KEY no Streamlit Secrets)
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", None)
if not OPENAI_KEY:
    st.error("OPENAI_API_KEY não encontrada em Streamlit Secrets. Vá em Manage App → Secrets e adicione.")
    st.stop()

CAMINHO_DB = "db"

# -------------------------
# Prompt seguro (não repetir histórico)
# -------------------------
# Observação: nós passamos o histórico ao prompt em formato *resumido* e damos instruções claras
# para NÃO repetir as marcações do histórico no texto de saída.
prompt_template = """
INSTRUÇÕES IMPORTANTES (LIMITE RÍGIDO):
- Você só pode responder usando EXCLUSIVAMENTE a "Base de conhecimento" fornecida abaixo.
- NÃO invente, NÃO adivinhe e NÃO use conhecimento externo.
- NÃO repita literalmente as marcações do histórico (por exemplo: "Usuário:", "Assistente:") na sua resposta.

Base de conhecimento (trechos recuperados):
{base_conhecimento}

Histórico resumido (apenas para contexto, NÃO repita marcações):
{historico}

Pergunta:
{pergunta}

Resposta (seja didático, explique causas e passos de solução com linguagem simples)
"""

# -------------------------
# Carregar modelos e DB
# -------------------------
@st.cache_resource
def carregar_modelos():
    # Embeddings: definimos explicitamente o modelo do embedding (ajuste se desejar)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model="text-embedding-3-small")
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embeddings)
    # Chat model: escolha um modelo disponível; gpt-4o-mini é sugerido como default
    modelo = ChatOpenAI(openai_api_key=OPENAI_KEY, model="gpt-4o-mini", temperature=0.2)
    return embeddings, db, modelo

embeddings, db, modelo = carregar_modelos()

# -------------------------
# Container do chat
# -------------------------
chat_box = st.container()

def render_chat(scroll=False):
    """Renderiza todo o histórico e o indicador 'digitando'."""
    with chat_box:
        st.markdown('<div id="chatbox" class="chat-container">', unsafe_allow_html=True)

        for troca in st.session_state["historico"]:
            # user message
            st.markdown(
                f"""
                <div class="msg-row user">
                    <div class="user-msg">{troca['user']}</div>
                    <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/9977/9977334.png">
                </div>
                """,
                unsafe_allow_html=True
            )

            # bot message (pode ser None ainda)
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

        # indicador "digitando..."
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

# Só injeta o JS de scroll quando explicitamente pedido (scroll=True)
    if scroll:
        st.markdown("""
        <script>
            const box = document.getElementById("chatbox");
            if (box) {
                // pequenas pausas ajudam o browser a estabilizar o layout antes de rolar
                setTimeout(() => { box.scrollTop = box.scrollHeight; }, 120);
            }
        </script>
        """, unsafe_allow_html=True)

# render inicial
# render inicial (sem scroll automático)
render_chat(scroll=False)

render_chat()
st.markdown('',unsafe_allow_html=True)

# -------------------------
# Input do usuário
# -------------------------
pergunta = st.chat_input("Digite sua dúvida...")

if pergunta:
    # 1) adicionar mensagem do usuário imediatamente (sem resposta)
    st.session_state["historico"].append({"user": pergunta, "bot": None})

    # 2) ativar o indicador de digitação e re-renderizar para o usuário ver "Digitando..."
    st.session_state["digitando"] = True
    render_chat()
    # Forçar scroll até o final (mostra a mensagem do usuário e o "Digitando...")
    #st.markdown("""
    #<script>
        #var box = document.getElementById("chatbox");
        #if (box) { box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' }); }
    #</script>
    #""", unsafe_allow_html=True)

    # 3) Agora geramos a resposta (bloqueante) — mantenha isso abaixo para garantir que o UX mostre "Digitando..."
    ultima_msg = pergunta

    # recuperar vetores/fragmentos
    vetor = embeddings.embed_query(ultima_msg)
    resultados = db.similarity_search_by_vector_with_relevance_scores(vetor, k=6)  # k maior para segurança

    # Filtra resultados (opcional): aqui usamos TODOS e avaliamos no prompt
    if not resultados or len(resultados) == 0:
        # sem dados no índice
        resposta_final = "Não encontrei informações suficientes na base de conhecimento para responder a isso."
    else:
        # coletar conteúdos (você pode limitar aqui tamanho/quantidade)
        # mantemos a ordem original; juntamos os page_content
        textos_resultado = []
        for doc, score in resultados:
            textos_resultado.append(doc.page_content)

        base_conhecimento = "\n\n----\n\n".join(textos_resultado)

        # preparar histórico resumido - sem marcações "Usuário/Assistente"
        # vamos passar apenas as últimas N trocas (ex.: 6) para evitar prompt muito grande
        resumo_historico = []
        for troca in st.session_state["historico"][:-1]:  # sem a mensagem atual
            if troca.get("bot"):
                resumo_historico.append(f"User: {troca['user']}\nAssistant: {troca['bot']}")
            else:
                resumo_historico.append(f"User: {troca['user']}")

        # limitar tamanho do histórico a N últimas entradas
        resumo_historico_text = "\n\n".join(resumo_historico[-6:])

        # montar prompt com instruções rígidas
        prompt = ChatPromptTemplate.from_template(prompt_template)
        prompt_injetado = prompt.invoke({
            "historico": resumo_historico_text,
            "base_conhecimento": base_conhecimento,
            "pergunta": ultima_msg
        })

        # gerar resposta a partir do modelo
        # Este é o ponto crítico — o prompt instrui fortemente para não "inventar"
        resposta_final = modelo.invoke(prompt_injetado).content

        # Se o modelo tentar burlar (por exemplo, responder algo muito curto ou genérico),
        # você pode checar aqui e forçar a resposta padrão. Exemplo:
        if not resposta_final or len(resposta_final.strip()) < 10:
            resposta_final = "Não encontrei informações suficientes na base de conhecimento para responder a isso."

    # 4) salvar resposta no histórico
    st.session_state["historico"][-1]["bot"] = resposta_final

    # 5) desativar indicador digitando e re-renderizar tudo com a resposta
    st.session_state["digitando"] = False
    st.rerun(scroll=True)

    
    # 6) scroll suave para o final para garantir que o usuário veja a resposta
    #st.markdown("""
    #<script>
        #var box = document.getElementById("chatbox");
        #if (box) { box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' }); }
    #</script>
    #""", unsafe_allow_html=True)
