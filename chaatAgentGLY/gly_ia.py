import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import requests
import groq

# Cargar variables del .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY: {'Set' if api_key else 'Not set'}")

# Función para generar el prompt según el rol y estilo
def generar_prompt(rol, estilo, consulta):
    estilos = {
        "Formal": "Usa un lenguaje profesional y directo.",
        "Conversacional": "Habla como si fueras un colega experto, con naturalidad.",
        "Técnico": "Incluye términos técnicos y explicaciones estructuradas."
    }
    introducciones = {
        "Auditor": "Actúas como un auditor empresarial especializado en detectar cuellos de botella y proponer automatizaciones con IA.",
        "Desarrollador": "Eres un desarrollador senior con experiencia en arquitecturas modernas, microservicios e IA aplicada.",
        "Gestor de Negocios": "Eres un estratega empresarial que busca oportunidades de eficiencia y escalabilidad.",
        "Investigador": "Tienes la misión de recopilar datos clave y proponer estrategias fundadas en datos reales."
    }
    
    return (
        f"{introducciones.get(rol, 'Eres un asistente de IA experto en empresas.')}\n"
        f"{estilos.get(estilo, '')}\n"
        f"Tu tarea es responder a esta consulta del usuario: \"{consulta}\"\n"
        f"Si el usuario no ha sido claro, guía la conversación con una pregunta concreta relacionada con su necesidad o sector."
    )

# Lógica principal
def gly_ia(query, rol="Auditor", temperatura=0.7, estilo="Formal"):
    try:
        if not api_key:
            raise ValueError("GROQ_API_KEY no está configurada")

        # Instanciar el modelo Llama 3 en Groq
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            api_key=api_key,
            temperature=float(temperatura),
            max_tokens=1200
        )

        # Generar prompt enriquecido
        prompt = generar_prompt(rol, estilo, query)

        # Obtener respuesta directamente
        respuesta = llm.invoke(prompt)

        return respuesta.content  # LangChain Message.content

    except groq.APIConnectionError as e:
        return f"❌ Error de conexión con Groq: {str(e)}"
    except groq.RateLimitError as e:
        return f"❌ Límite de la API alcanzado: {str(e)}"
    except groq.AuthenticationError as e:
        return f"❌ Error de autenticación: Clave de API inválida - {str(e)}"
    except Exception as e:
        return f"❌ Error inesperado: {str(e)}"

# CLI para pruebas por consola
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python gly_ia.py '{query}' '{rol}' '{temperatura}' '{estilo}'")
        sys.exit(1)

    query = sys.argv[1]
    rol = sys.argv[2] if len(sys.argv) > 2 else "Auditor"
    temperatura = sys.argv[3] if len(sys.argv) > 3 else 0.7
    estilo = sys.argv[4] if len(sys.argv) > 4 else "Formal"

    print("\n=== GLY-IA está generando la respuesta... ===\n")
    salida = gly_ia(query, rol, temperatura, estilo)
    print("\n=== RESPUESTA DE GLY-IA ===\n")
    print(salida)
