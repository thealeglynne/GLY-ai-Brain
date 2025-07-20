import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import groq

# ===== Cargar entorno =====
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY: {'Set' if api_key else 'Not set'}")

# ===== Generar instrucciones por rol y estilo =====
def generar_instrucciones(rol, estilo):
    estilos = {
        "Formal": "Usa un lenguaje profesional, claro y directo.",
        "Conversacional": "Habla como un colega experto, con naturalidad, amabilidad y claridad, como si conversaras en persona.",
        "Técnico": "Usa términos técnicos y explicaciones estructuradas, manteniendo claridad."
    }

    introducciones = {
        "Auditor": (
            "Eres GLY-IA, un consultor experto en auditorías empresariales y transformación digital. Tu misión es recolectar información clave sobre los procesos, herramientas y desafíos del usuario para preparar una auditoría técnica. Haz preguntas específicas, una por respuesta, para entender su negocio y detectar ineficiencias. Mantén las respuestas breves (80-120 palabras), naturales y basadas en el contexto, evitando explicaciones largas o repeticiones. Propón escribir 'generar auditoria' cuando tengas suficiente información (por ejemplo, tras 3-4 interacciones relevantes). Tu objetivo es guiar al usuario hacia una auditoría sin abrumarlo, proponiendo soluciones basadas en IA solo cuando sea necesario."
        ),
        "Desarrollador": "Eres un desarrollador senior con experiencia en arquitecturas modernas, microservicios e IA aplicada.",
        "Gestor de Negocios": "Eres un estratega empresarial que busca oportunidades de eficiencia y escalabilidad.",
        "Investigador": "Tienes la misión de recopilar datos clave y proponer estrategias basadas en datos reales."
    }

    return f"{introducciones.get(rol, 'Eres un asistente de IA experto en empresas.')}\n{estilos.get(estilo, '')}"

# ===== Construcción de historial para el prompt =====
def construir_contexto(historial, max_turnos=5):
    if len(historial) > max_turnos:
        historial = historial[-max_turnos:]
    contexto = "\n".join([
        f"Usuario: {turno['user']}\nIA: {turno['ia']}" for turno in historial
        if turno['user'].lower() != "iniciar conversación"
    ])
    return contexto

# ===== Evaluar si ya se puede generar auditoría =====
def evaluar_completitud(historial):
    criterios = [
        "proceso", "herramienta", "problema", "ineficiencia", "flujo"
    ]
    texto_completo = " ".join([turno["user"].lower() for turno in historial])
    return sum(criterio in texto_completo for criterio in criterios) >= 2 and len(historial) >= 3

# ===== Guardar conversación en JSON =====
def guardar_conversacion_json(historial, empresa="Desconocida", rol="Auditor", estilo="Conversacional"):
    data = {
        "empresa": empresa,
        "rol": rol,
        "estilo": estilo,
        "fecha": datetime.now().isoformat(),
        "conversacion": historial,
        "ready_to_generate": evaluar_completitud(historial)
    }

    with open("conversacion_gly_ia.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ===== Prompt Template =====
prompt_template = PromptTemplate(
    input_variables=["instrucciones", "contexto", "input"],
    template=(
        "{instrucciones}\n\n"
        "Contexto reciente de la conversación:\n{contexto}\n\n"
        "Nueva pregunta del usuario: {input}\n\n"
        "Responde de forma breve (80-120 palabras), clara y natural, como si hablaras con un colega. Usa el contexto para continuar la conversación de forma lógica, sin repetir ideas. Haz una sola pregunta clave para recolectar información sobre procesos, herramientas o desafíos. Evita explicaciones largas o sugerencias prematuras de soluciones. Si el contexto es suficiente, sugiere escribir 'generar auditoria' de forma breve. No repitas la pregunta del usuario en la respuesta.\n\n"
        "Respuesta de GLY-IA:"
    )
)

# ===== Llamada principal del agente =====
def gly_ia(query, rol="Auditor", temperatura=0.7, estilo="Conversacional", historial=None):
    try:
        if not api_key:
            raise ValueError("GROQ_API_KEY no está configurada")

        if historial is None:
            historial = []

        # Manejar el query inicial
        if query.lower() == "iniciar conversación":
            respuesta = (
                "¡Hola! Soy GLY-IA, tu asistente para auditar procesos con IA. 😊 Quiero entender tu negocio. ¿A qué se dedica tu empresa?"
            )
            historial.append({"user": query, "ia": respuesta})
            guardar_conversacion_json(historial, rol=rol, estilo=estilo)
            return respuesta, historial

        instrucciones = generar_instrucciones(rol, estilo)
        contexto = construir_contexto(historial)

        prompt = prompt_template.format(
            instrucciones=instrucciones,
            contexto=contexto,
            input=query
        )

        llm = ChatGroq(
            model_name="llama3-70b-8192",
            api_key=api_key,
            temperature=float(temperatura),
            max_tokens=150  # Reducido para respuestas más cortas
        )

        respuesta = llm.invoke(prompt)
        texto = respuesta.content if hasattr(respuesta, "content") else str(respuesta)

        # === Agregar sugerencia de auditoría si el contexto es suficiente ===
        if evaluar_completitud(historial) and "generar auditoria" not in query.lower():
            texto += "\n\nParece que tenemos suficiente info. ¿Listo para el informe técnico? Escribe 'generar auditoria'."

        historial.append({"user": query, "ia": texto})

        # === Guardar conversación en JSON ===
        guardar_conversacion_json(historial, rol=rol, estilo=estilo)

        return texto, historial

    except groq.APIConnectionError as e:
        return f"❌ Error de conexión con Groq: {str(e)}", historial
    except groq.RateLimitError as e:
        return f"❌ Límite de la API alcanzado: {str(e)}", historial
    except groq.AuthenticationError as e:
        return f"❌ Error de autenticación: Clave de API inválida - {str(e)}", historial
    except Exception as e:
        return f"❌ Error inesperado: {str(e)}", historial

# ===== CLI para pruebas rápidas =====
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python gly_ia.py '{query}' '{rol}' '{temperatura}' '{estilo}'")
        sys.exit(1)

    query = sys.argv[1]
    rol = sys.argv[2] if len(sys.argv) > 2 else "Auditor"
    temperatura = sys.argv[3] if len(sys.argv) > 3 else 0.7
    estilo = sys.argv[4] if len(sys.argv) > 4 else "Conversacional"

    print("\n=== GLY-IA está generando la respuesta... ===\n")

    historial_chat = []
    salida, historial_chat = gly_ia(query, rol, temperatura, estilo, historial=historial_chat)

    print("\n=== RESPUESTA DE GLY-IA ===\n")
    print(salida)