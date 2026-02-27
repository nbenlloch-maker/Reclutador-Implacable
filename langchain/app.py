import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

# --- 1. CONFIGURACIÓN VISUAL DE LA PÁGINA ---
st.set_page_config(page_title="Simulador de Entrevistas AI", page_icon="👔", layout="centered")

# --- 2. BARRA LATERAL (SIDEBAR) ESTÉTICA ---
with st.sidebar:
    st.title("⚙️ Configuración")
    st.markdown("Bienvenido al simulador. Para que el reclutador AI comience a evaluarte, necesitamos conectar tu cuenta de Google.")
    
    # Input de API Key con estilo
    api_key = st.text_input("🔑 Ingresa tu Google API Key:", type="password", help="Consigue tu API key gratuita en Google AI Studio.")
    
    st.divider()
    
    # Un toque extra: Elegir el rol para personalizar la experiencia
    st.markdown("### 🎯 Detalles de la Pasantía")
    rol = st.selectbox("¿A qué área estás aplicando?", 
                       ["Desarrollo de Software", "Marketing Digital", "Análisis de Datos", "Finanzas", "Recursos Humanos"])
    
    st.divider()
    if st.button("🔄 Reiniciar Entrevista", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 3. PANTALLA PRINCIPAL ---
st.title("👔 El Reclutador Implacable")
st.markdown(f"**Entrevista para Pasantía en:** `{rol}`")
st.markdown("Prepárate. Nuestro reclutador de IA detecta respuestas genéricas y clichés. Te presionará para que des ejemplos reales y métricas de impacto.")
st.divider()

# Detener la app visualmente si no hay API Key
if not api_key:
    st.warning("👈 Por favor, ingresa tu API Key de Google en el menú lateral para iniciar la simulación.")
    st.stop()

# --- 4. INICIALIZACIÓN DE MEMORIA Y ESTADO ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # El bot da el primer paso de forma proactiva
    pregunta_inicial = f"Hola. Veo que aplicas a la pasantía de {rol}. Para empezar, háblame de un proyecto difícil que hayas sacado adelante y qué rol exacto jugaste tú."
    st.session_state.current_question = pregunta_inicial
    st.session_state.messages.append({"role": "assistant", "content": pregunta_inicial})

# --- 5. LÓGICA DE LANGCHAIN (Basada en tu ejercicio) ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.6)
output_parser = StrOutputParser()

# Clasificador dinámico
classifier_template = PromptTemplate(
    input_variables=["question", "answer"],
    template="""Eres un evaluador estricto. Analiza la respuesta a la pregunta.
    Pregunta: {question}
    Respuesta: {answer}
    Si la respuesta da ejemplos concretos, menciona tecnologías/herramientas o detalla el 'cómo', clasifícala como 'Fuerte'.
    Si la respuesta usa clichés, es muy teórica, vaga o le falta detalle, clasifícala como 'Débil'.
    Responde ÚNICAMENTE con la palabra: Fuerte o Débil.
    Clasificación:"""
)
classifier_chain = classifier_template | llm | output_parser #

# Respuestas enrutadas
strong_template = PromptTemplate(
    input_variables=["answer"],
    template="""El candidato respondió: '{answer}'.
    Valida su respuesta brevemente (1 línea) y hazle una NUEVA pregunta técnica o de comportamiento más difícil sobre lo que acaba de mencionar.
    Nueva Pregunta:"""
)
strong_chain = strong_template | llm | output_parser #

weak_template = PromptTemplate(
    input_variables=["answer"],
    template="""El candidato respondió: '{answer}'.
    Dile directamente y con tono profesional por qué su respuesta es insuficiente (muy general, sin ejemplos). EXÍGELE que te dé un ejemplo concreto de su vida académica o laboral que demuestre esa habilidad.
    Tu respuesta:"""
)
weak_chain = weak_template | llm | output_parser #

# Enrutamiento dinámico
def route(info):
    if "fuerte" in info["result"].strip().lower():
        return strong_chain
    else:
        return weak_chain

routing_chain = (
    RunnableParallel({"result": classifier_chain, "question": lambda x: x["question"], "answer": lambda x: x["answer"]}) #
    | RunnableLambda(route) #
)

# --- 6. RENDERIZADO DEL CHAT (INTERFAZ VISUAL) ---
# Mostramos el historial con avatares atractivos
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🧑‍💼"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])

# --- 7. CAJA DE INPUT DEL USUARIO ---
user_input = st.chat_input("Escribe tu respuesta aquí detalladamente...")

if user_input:
    # Imprimir lo que dice el usuario en pantalla
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # El bot piensa y responde
    with st.chat_message("assistant", avatar="🧑‍💼"):
        with st.spinner("Evaluando tu respuesta..."):
            
            response = routing_chain.invoke({
                "question": st.session_state.current_question,
                "answer": user_input
            }) #
            
            st.markdown(response)
            
            # Guardamos en memoria
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.current_question = response