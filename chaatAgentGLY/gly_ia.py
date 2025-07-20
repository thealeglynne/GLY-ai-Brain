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
supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "https://zixyjbmaczqsitxubcbp.supabase.co")
supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")

print(f"GROQ_API_KEY: {'Set' if api_key else 'Not set'}")
print(f"Supabase URL: {supabase_url}")

# ===== Inicializar cliente de Supabase =====
from supabase import create_client, Client
supabase: Client = create_client(supabase_url, supabase_key)

# ===== Recuperar auditorías previas del usuario =====
async def get_user_audits(user_id, max_turnos=5):
    try:
        response = await supabase \
            .from_('auditorias') \
            .select('audit_content') \
            .eq('user_id', user_id) \
            .order('created_at', desc=True) \
            .limit(max_turnos) \
            .execute()
        
        if response.data is None:
            print(f"❌ No se encontraron auditorías para el usuario")
            return []
        
        audits = []
        for audit in response.data:
            try:
                audits.append(json.loads(audit['audit_content']))
            except Exception:
                continue

        return audits
    except Exception as e:
        print(f"❌ Error al obtener auditorías: {str(e)}")
        return []

# ===== Generar instrucciones por rol y estilo =====
def generar_instrucciones(rol, estilo):
    estilos = {
        "Formal": "Usa un lenguaje profesional, claro y directo.",
        "Conversacional": "Habla como un colega experto, con naturalidad, amabilidad y claridad, como si conversaras en persona.",
        "Técnico": "Usa términos técnicos y explicaciones estructuradas, manteniendo claridad."
    }

    introducciones = {
        "Auditor": (
            "Eres GLY-IA, un consultor experto en auditorías empresariales y transformación digital. Tu misión es recolectar información clave sobre los procesos, herramientas y desafíos del usuario para preparar una auditoría técnica. Usa el historial de auditorías previas del usuario para contextualizar tus respuestas. Haz preguntas específicas, una por respuesta, para entender su negocio y detectar ineficiencias. Mantén las respuestas breves (80-120 palabras), naturales y basadas en el contexto, evitando explicaciones largas o repeticiones. Propón escribir 'generar auditoria' cuando tengas suficiente información (por ejemplo, tras 3-4 interacciones relevantes). Tu objetivo es guiar al usuario hacia una auditoría sin abrumarlo, proponiendo soluciones basadas en IA solo cuando sea necesario."
        ),
        "Desarrollador": "Eres un desarrollador senior con experiencia en arquitecturas modernas, microservicios e IA aplicada.",
        "Gestor de Negocios": "Eres un estratega empresarial que busca oportunidades de eficiencia y escalabilidad.",
        "Investigador": "Tienes la misión de recopilar datos clave y proponer estrategias basadas en datos reales."
    }

    return f"{introducciones.get(rol, 'Eres un asistente de IA experto en empresas.')}\n{estilos.get(estilo, '')}"

# ===== Construcción de contexto desde auditorías =====
async def construir_contexto(user_id, max_turnos=5):
    audits = await get_user_audits(user_id, max_turnos)
    contexto = ""
    for audit in audits:
        user_input = audit.get('user_input', '')
        ia_response = audit.get('ia_response', '')
        if user_input and ia_response:
            contexto += f"Usuario: {user_input}\nIA: {ia_response}\n"
    return contexto

# ===== Evaluar si se puede generar auditoría =====
def evaluar_completitud(audits):
    criterios = ["proceso", "herramienta", "problema", "ineficiencia", "flujo"]
    texto_completo = " ".join([audit.get('user_input', '').lower() for audit in audits])
    return sum(criterio in texto_completo for criterio in criterios) >= 2 and len(audits) >= 3

# ===== Prompt Template =====
prompt_template = PromptTemplate(
    input_variables=["instrucciones", "contexto", "input"],
    template=(
        "{instrucciones}\n\n"
        "Contexto reciente de la conversación (basado en auditorías previas del usuario):\n{contexto}\n\n"
        "Nueva pregunta del usuario: {input}\n\n"
        "Responde de forma breve (80-120 palabras), clara y natural, como si hablaras con un colega. "
        "Usa el contexto para continuar la conversación de forma lógica, sin repetir ideas. "
        "Haz una sola pregunta clave para recolectar información sobre procesos, herramientas o desafíos. "
        "Evita explicaciones largas o sugerencias prematuras de soluciones. "
        "Si el contexto es suficiente, sugiere escribir 'generar auditoria' de forma breve. "
        "No repitas la pregunta del usuario en la respuesta.\n\n"
        "Respuesta de GLY-IA:"
    )
)

# ===== Función principal del agente =====
async def gly_ia(query, user_id, rol="Auditor", temperatura=0.7, estilo="Conversacional"):
    try:
        if not api_key:
            raise ValueError("GROQ_API_KEY no está configurada")

        if not user_id:
            return "❌ No se recibió un user_id válido.", []

        audits = await get_user_audits(user_id)
        instrucciones = generar_instrucciones(rol, estilo)
        contexto = await construir_contexto(user_id)

        prompt = prompt_template.format(
            instrucciones=instrucciones,
            contexto=contexto,
            input=query
        )

        llm = ChatGroq(
            model_name="llama3-70b-8192",
            api_key=api_key,
            temperature=float(temperatura),
            max_tokens=150
        )

        respuesta = llm.invoke(prompt)
        texto = respuesta.content if hasattr(respuesta, "content") else str(respuesta)

        if evaluar_completitud(audits) and "generar auditoria" not in query.lower():
            texto += "\n\nParece que tenemos suficiente info. ¿Listo para el informe técnico? Escribe 'generar auditoria'."

        return texto, audits

    except groq.APIConnectionError as e:
        return f"❌ Error de conexión con Groq: {str(e)}", []
    except groq.RateLimitError as e:
        return f"❌ Límite de la API alcanzado: {str(e)}", []
    except groq.AuthenticationError as e:
        return f"❌ Error de autenticación: Clave de API inválida - {str(e)}", []
    except Exception as e:
        return f"❌ Error inesperado: {str(e)}", []

# ===== CLI para pruebas rápidas =====
if __name__ == "__main__":
    import asyncio

    if len(sys.argv) < 3:
        print("Uso: python gly_ia.py '{query}' '{user_id}' '{rol}' '{temperatura}' '{estilo}'")
        sys.exit(1)

    query = sys.argv[1]
    user_id = sys.argv[2]
    rol = sys.argv[3] if len(sys.argv) > 3 else "Auditor"
    temperatura = sys.argv[4] if len(sys.argv) > 4 else 0.7
    estilo = sys.argv[5] if len(sys.argv) > 5 else "Conversacional"

    print("\n=== GLY-IA está generando la respuesta... ===\n")

    async def main():
        salida, audits = await gly_ia(query, user_id, rol, temperatura, estilo)
        print("\n=== RESPUESTA DE GLY-IA ===\n")
        print(salida)

    asyncio.run(main())
