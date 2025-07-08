import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Cargar variables del .env
load_dotenv()

# Obtener clave desde .env
api_key = os.getenv("GROQ_API_KEY")

# Función para generar prompt según rol y estilo
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
        f"{introducciones.get(rol, 'Eres un asistente de IA experto en empresas.')} "
        f"{estilos.get(estilo, '')} "
        f"Tu tarea es responder a esta consulta del usuario: \"{consulta}\". "
        f"Si el usuario no ha sido claro, guía la conversación con una pregunta concreta relacionada con su necesidad o sector."
    )

# Lógica principal
def gly_ia(query, rol="Auditor", temperatura=0.7, estilo="Formal"):
    try:
        # Instanciar el modelo desde Groq
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            api_key=api_key,
            temperature=float(temperatura),
            max_tokens=1200
        )

        prompt = generar_prompt(rol, estilo, query)

        prompt_template = PromptTemplate(
            input_variables=["query", "prompt"],
            template="{prompt}"
        )

        chain = LLMChain(llm=llm, prompt=prompt_template)

        respuesta = chain.invoke({"query": query, "prompt": prompt})
        return respuesta

    except Exception as e:
        return f"❌ Error al procesar la consulta: {str(e)}"

# CLI
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
    print(salida if isinstance(salida, str) else salida.get("text", "Sin respuesta."))
