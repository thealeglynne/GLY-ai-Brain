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

# ===== Guardar conversación en JSON (solo para depuración) =====
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

def gly_ia(query, rol="Auditor", temperatura=0.7, estilo="Conversacional", historial=None):
    try:
        if not api_key:
            raise ValueError("GROQ_API_KEY no está configurada")

        # --- Inicio: Optimización de tokens ---
        # Limitar longitud de la consulta
        query = query[:400]  # Máximo 400 caracteres por mensaje
        
        # Limitar historial a los últimos 3 intercambios (reduce tokens)
        historial = historial[-3:] if historial else []
        
        # Acortar instrucciones para ahorrar tokens
        instrucciones = generar_instrucciones(rol, estilo)[:600]  # Máx 600 caracteres
        # --- Fin: Optimización de tokens ---

        # Reiniciar la conversación si es una nueva sesión
        if query.lower() == "iniciar conversación":
            respuesta = (
                "¡Hola! Soy GLY-IA, tu asistente para auditar procesos con IA. 😊 Quiero entender tu negocio. ¿A qué se dedica tu empresa?"
            )
            return respuesta, [{"user": query, "ia": respuesta}]

        # Manejar el comando para generar auditoría
        if query.strip().lower() == "generar auditoria":
            return "✅ Auditoría finalizada. Propuesta técnica generada.", []

        # Construir contexto optimizado (limitar longitud)
        contexto = "\n".join([
            f"U: {turno['user'][:150]}\nIA: {turno['ia'][:150]}"  # Limitar a 150 chars por turno
            for turno in historial
            if turno['user'].lower() != "iniciar conversación"
        ])

        prompt = prompt_template.format(
            instrucciones=instrucciones,
            contexto=contexto,
            input=query[:500]  # Limitar input del usuario
        )

        # Configuración optimizada de Groq
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            api_key=api_key,
            temperature=float(temperatura),
            max_tokens=180,  # Aumentado ligeramente a 180 tokens
            request_timeout=12  # Timeout de 12 segundos
        )

        respuesta = llm.invoke(prompt)
        texto = respuesta.content[:400] if hasattr(respuesta, "content") else str(respuesta)[:400]

        # Agregar sugerencia de auditoría si el contexto es suficiente
        if evaluar_completitud(historial) and "generar auditoria" not in query.lower():
            texto += "\n\n[¿Listo para generar el informe técnico? Escribe 'generar auditoria']"

        # Actualizar historial (manteniendo máximo 4 intercambios)
        nuevo_historial = (historial + [{"user": query[:300], "ia": texto}])[-4:]
        
        # Guardar conversación (opcional para depuración)
        guardar_conversacion_json(nuevo_historial, rol=rol, estilo=estilo)

        return texto, nuevo_historial

    except groq.APIConnectionError as e:
        return "🔌 Error de conexión. Por favor intenta nuevamente.", historial
    except groq.RateLimitError as e:
        return "⚠️ Límite temporal de la API. Espera 1 minuto.", historial
    except groq.APIError as e:
        if e.code == 503:
            return "⏳ Servicio ocupado. Intenta nuevamente en breve.", historial
        return f"❌ Error en la API: {str(e)[:200]}", historial
    except Exception as e:
        return f"⚠️ Error: {str(e)[:200]}", historial
    
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

    # Para pruebas CLI, mantenemos un historial simple en memoria
    if not hasattr(gly_ia, 'historial_cli'):
        gly_ia.historial_cli = []

    salida, gly_ia.historial_cli = gly_ia(query, rol, temperatura, estilo, historial=gly_ia.historial_cli)

    print("\n=== RESPUESTA DE GLY-IA ===\n")
    print(salida)