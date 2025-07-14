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
        "Formal": "Usa un lenguaje profesional y directo.",
        "Conversacional": "Habla como si fueras un colega experto, con naturalidad.",
        "Técnico": "Incluye términos técnicos y explicaciones estructuradas."
    }

    introducciones = {
        "Auditor": (
            "Actúas como un auditor empresarial y arquitecto de software especializado en detectar cuellos de botella, "
            "diseñar ciudades digitales y proponer arquitecturas escalables donde la inteligencia artificial no solo "
            "automatiza, sino habita y transforma. Consideras los procesos como avenidas por donde fluyen datos, "
            "y los modelos de IA como entidades inteligentes que trabajan dentro de servicios modulares, microservicios "
            "y capas de orquestación. Tu enfoque va más allá de chatbots: propones sistemas vivos que se acoplan de forma "
            "nativa a los flujos de trabajo, impulsados por IA y diseñados para evolucionar con la empresa."
        ),
        "Desarrollador": "Eres un desarrollador senior con experiencia en arquitecturas modernas, microservicios e IA aplicada.",
        "Gestor de Negocios": "Eres un estratega empresarial que busca oportunidades de eficiencia y escalabilidad.",
        "Investigador": "Tienes la misión de recopilar datos clave y proponer estrategias fundadas en datos reales."
    }

    return f"{introducciones.get(rol, 'Eres un asistente de IA experto en empresas.')}\n{estilos.get(estilo, '')}"

# ===== Construcción de historial para el prompt =====
def construir_contexto(historial, max_turnos=3):
    if len(historial) > max_turnos:
        historial = historial[-max_turnos:]
    contexto = "\n".join([
        f"Usuario: {turno['user']}\nIA: {turno['ia']}" for turno in historial
    ])
    return contexto

# ===== Evaluar si ya se puede generar auditoría =====
def evaluar_completitud(historial):
    criterios = [
        "automatización", "proceso", "departamento", "herramienta",
        "problema", "flujo", "ineficiencia"
    ]
    texto_completo = " ".join([turno["user"].lower() for turno in historial])
    return all(criterio in texto_completo for criterio in criterios) or len(historial) >= 6

# ===== Guardar conversación en JSON =====
def guardar_conversacion_json(historial, empresa="Desconocida", rol="Auditor", estilo="Formal"):
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
        "Contexto reciente:\n{contexto}\n\n"
        "Nueva entrada del usuario: {input}\n"
        "IA:"
    )
)

# ===== Llamada principal del agente =====
def gly_ia(query, rol="Auditor", temperatura=0.7, estilo="Formal", historial=None):
    try:
        if not api_key:
            raise ValueError("GROQ_API_KEY no está configurada")

        if historial is None:
            historial = []

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
            max_tokens=1000
        )

        respuesta = llm.invoke(prompt)
        texto = respuesta.content if hasattr(respuesta, "content") else str(respuesta)

        # === Agregar sugerencia a partir del séptimo mensaje ===
        if len(historial) >= 6 and "generar auditoria" not in query.lower():
            texto += (
                "\n\n🧠 Parece que ya tenemos un contexto muy completo. "
                "Cuando estés listo, escribe **'generar auditoria'** para que pueda producir el informe técnico consultivo."
            )

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
    estilo = sys.argv[4] if len(sys.argv) > 4 else "Formal"

    print("\n=== GLY-IA está generando la respuesta... ===\n")

    historial_chat = []
    salida, historial_chat = gly_ia(query, rol, temperatura, estilo, historial=historial_chat)

    print("\n=== RESPUESTA DE GLY-IA ===\n")
    print(salida)
