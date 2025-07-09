import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import groq

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY: {'Set' if api_key else 'Not set'}")

# Función para generar instrucciones del sistema según rol y estilo
def generar_instrucciones(rol, estilo):
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
    return f"{introducciones.get(rol, 'Eres un asistente de IA experto en empresas.')}\n{estilos.get(estilo, '')}"

# Prompt con historial incluido
prompt_template = PromptTemplate(
    input_variables=["history", "input", "instrucciones"],
    template=(
        "{instrucciones}\n"
        "Historial de conversación:\n{history}\n"
        "Usuario: {input}\n"
        "IA:"
    )
)

# Lógica principal
def gly_ia(query, rol="Auditor", temperatura=0.7, estilo="Formal", memory=None):
    try:
        if not api_key:
            raise ValueError("GROQ_API_KEY no está configurada")

        instrucciones = generar_instrucciones(rol, estilo)

        llm = ChatGroq(
            model_name="llama3-70b-8192",
            api_key=api_key,
            temperature=float(temperatura),
            max_tokens=1200
        )

        if memory is None:
            memory = ConversationBufferMemory(return_messages=True)

        chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            memory=memory,
            verbose=False
        )

        respuesta = chain.run({
            "input": query,
            "instrucciones": instrucciones
        })

        return respuesta, memory

    except groq.APIConnectionError as e:
        return f"❌ Error de conexión con Groq: {str(e)}", memory
    except groq.RateLimitError as e:
        return f"❌ Límite de la API alcanzado: {str(e)}", memory
    except groq.AuthenticationError as e:
        return f"❌ Error de autenticación: Clave de API inválida - {str(e)}", memory
    except Exception as e:
        return f"❌ Error inesperado: {str(e)}", memory

# CLI para pruebas
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python gly_ia.py '{query}' '{rol}' '{temperatura}' '{estilo}'")
        sys.exit(1)

    query = sys.argv[1]
    rol = sys.argv[2] if len(sys.argv) > 2 else "Auditor"
    temperatura = sys.argv[3] if len(sys.argv) > 3 else 0.7
    estilo = sys.argv[4] if len(sys.argv) > 4 else "Formal"

    print("\n=== GLY-IA está generando la respuesta... ===\n")

    # Usamos una memoria persistente para mantener el hilo
    mem = ConversationBufferMemory(return_messages=True)
    salida, mem = gly_ia(query, rol, temperatura, estilo, memory=mem)

    print("\n=== RESPUESTA DE GLY-IA ===\n")
    print(salida)

    # Si quieres ver historial acumulado:
    # print("\n=== HISTORIAL DE CONVERSACIÓN ===\n")
    # print(mem.buffer)
