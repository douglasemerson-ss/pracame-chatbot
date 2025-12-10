import streamlit as st
import time

# =============================
#   CONFIGURAÇÃO DO APP
# =============================
st.set_page_config(page_title="Chat RAG", layout="wide")

# =============================
#   CSS PARA DEIXAR TIPO CHAT
# =============================
st.markdown("""
<style>
.chat-container {
    max-height: 600px;
    overflow-y: auto;
    padding-right: 10px;
}

.user-message {
    background-color: #DCF8C6;
    padding: 10px 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    max-width: 80%;
    align-self: flex-end;
}

.bot-message {
    background-color: #ECECEC;
    padding: 10px 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    max-width: 80%;
    align-self: flex-start;
}

.message-container {
    display: flex;
    flex-direction: column;
}
</style>
""", unsafe_allow_html=True)

# =============================
#   ESTADO DAS MENSAGENS
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =============================
#   FUNÇÃO: EFEITO DE DIGITAÇÃO
# =============================
def typewriter(text, speed=0.02):
    """Simula efeito de digitação."""
    typed = ""
    for char in text:
        typed += char
        yield typed
        time.sleep(speed)


# =============================
#   EXIBIÇÃO DO CHAT
# =============================
st.markdown("## Chat RAG")

chat_box = st.container()
with chat_box:
    st.markdown('<div class="chat-container" id="chat">', unsafe_allow_html=True)

    st.markdown('<div class="message-container">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{msg["content"]}</div>',
                        unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# =============================
#   INPUT DO USUÁRIO
# =============================
pergunta = st.chat_input("Digite sua pergunta...")

if pergunta:
    # salva mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": pergunta})

    # ---------------------------
    #   SUA LÓGICA DE RAG AQUI
    # ---------------------------
    # resposta = gerar_resposta(pergunta)
    resposta = gerar_resposta(pergunta)   # <- coloque sua função real

    # salva placeholder para animação
    placeholder = st.empty()

    texto_parcial = ""

    # efeito digitando
    for parte in typewriter(resposta):
        texto_parcial = parte
        placeholder.markdown(
            f'<div class="bot-message">{texto_parcial}</div>',
            unsafe_allow_html=True
        )
        # scroll automático
        st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>",
                     unsafe_allow_html=True)

    # salva mensagem final
    st.session_state.messages.append({"role": "assistant", "content": resposta})

    # força atualização para fixar mensagem
    st.rerun()
