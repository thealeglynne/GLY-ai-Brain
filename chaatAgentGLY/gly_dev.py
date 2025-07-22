# gly_dev.py

import os
import sys
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import groq

# ===== Cargar entorno =====
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY: {'Set' if api_key else 'Not set'}")

# ===== Cargar JSON de la conversación previa =====
def cargar_conversacion():
    try:
        with open("conversacion_gly_ia.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error al cargar el JSON de conversación: {e}") 
        return None

# ===== Instrucción para el generador técnico-consultivo =====
def generar_instrucciones_especializadas():
    return (
        "Actúas como un consultor senior de automatización de procesos y transformación digital en GLYNNE. "
        "Tu función es analizar a fondo el caso del cliente, contenido en el archivo JSON, y elaborar un documento consultivo de alto nivel. "
        "Este documento debe detallar cómo un enfoque integral de automatización, basado en inteligencia artificial y rediseño de procesos, puede resolver eficazmente los problemas descritos por el cliente.\n\n"
        "La propuesta no debe enfocarse en tecnologías específicas como lenguajes o frameworks. En su lugar, explica cómo automatizar los flujos puede resolver los cuellos de botella, mejorar la productividad, profesionalizar la operación y preparar a la empresa para escalar.\n\n"
        "Incluye secciones como: Diagnóstico Estratégico, Solución Propuesta, Beneficios Esperados, y Recomendaciones para un Análisis Más Profundo. El lenguaje debe ser claro, profesional, enfocado en impacto y resultados. No es un pitch comercial ni técnico, es un informe de consultoría especializado.\n\n"
        "Usa al menos 10 párrafos con redacción fluida. El documento debe poder leerse por un gerente general o un líder de transformación digital."
    )

# ===== Prompt Template =====
prompt_template = PromptTemplate(
    input_variables=["instrucciones", "contenido_json"],
    template=(
        "{instrucciones}\n\n"
        "Este es el contenido del análisis del cliente realizado previamente:\n"
        "{contenido_json}\n\n"
        "Redacta el documento consultivo:"
    )
)

def generar_documento_consultivo():
    try:
        if not api_key:
            raise ValueError("GROQ_API_KEY no está configurada")

        data = cargar_conversacion()
        if not data:
            return "❌ No se pudo cargar el archivo JSON de conversación."

        instrucciones = generar_instrucciones_especializadas()

        # --- NUEVO: Acortar contenido JSON si es muy extenso ---
        contenido_json = json.dumps(data, ensure_ascii=False)[:10000]  # Máx 10k caracteres

        prompt = prompt_template.format(
            instrucciones=instrucciones,
            contenido_json=contenido_json
        )

        # --- NUEVO: Ajustes para documentos largos ---
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            api_key=api_key,
            temperature=0.5,
            max_tokens=3500,  # Reducido para evitar error 413
            request_timeout=30  # Más tiempo para prompts largos
        )

        respuesta = llm.invoke(prompt)
        return respuesta.content if hasattr(respuesta, "content") else str(respuesta)

    except Exception as e:
        return f"❌ Error inesperado: {str(e)}"


# ===== CLI de prueba =====
if __name__ == "__main__":
    print("\n=== GLY-DEV está generando el informe consultivo especializado... ===\n")
    resultado = generar_documento_consultivo()
    print("\n=== INFORME CONSULTIVO ESPECIALIZADO ===\n")
    print(resultado)
