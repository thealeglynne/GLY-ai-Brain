from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# --- Importar funciones ---
try:
    from chaatAgentGLY.gly_ia import gly_ia
    from chaatAgentGLY.gly_dev import generar_documento_consultivo
except ImportError as e:
    print("❌ Error al importar agentes:", e)
    raise

# --- Inicializar FastAPI ---
app = FastAPI()

# --- Configurar CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gly-ai-arq.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Estado Global (simulación temporal para sesiones) ---
historial_global = []

# --- Entrada esperada desde el frontend ---
class ConsultaInput(BaseModel):
    query: str
    rol: str = "Auditor"
    temperatura: float = 0.7
    estilo: str = "Formal"

# --- Endpoint principal del agente GLY-IA ---
@app.post("/gpt")
async def procesar_consulta(data: ConsultaInput):
    if not data.query.strip():
        raise HTTPException(status_code=400, detail="El campo 'query' no puede estar vacío.")
    
    if len(data.query) > 300:
        raise HTTPException(
            status_code=400,
            detail="La consulta excede 300 caracteres. Por favor sé más breve."
        )

    try:
        global historial_global

        # Detectar trigger para generar propuesta técnica
        if data.query.strip().lower() == "generar auditoria":
            propuesta = generar_documento_consultivo()
            return {
                "respuesta": "✅ Auditoría finalizada. Propuesta técnica generada.",
                "propuesta": propuesta
            }

        # Caso normal: continuar conversación con GLY-IA
        respuesta, historial_global = gly_ia(
            data.query,
            rol=data.rol,
            estilo=data.estilo,
            temperatura=data.temperatura,
            historial=historial_global
        )

        return {"respuesta": respuesta}

    except Exception as e:
        print("❌ Error interno:", str(e))
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# --- Endpoint adicional para generar propuesta manual ---
@app.get("/propuesta-tecnica")
async def generar_propuesta():
    try:
        propuesta = generar_documento_consultivo()
        return {"propuesta": propuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar propuesta: {str(e)}")

# --- Nuevo endpoint para monitoreo ---
@app.post("/estado-tokens")
async def estado_tokens():
    """Devuelve el estado actual del uso de tokens"""
    return {
        "limites": {
            "conversacion": {
                "max_tokens": 120,
                "max_historial": 3,
                "max_query_chars": 300
            },
            "documentos": {
                "max_tokens": 3500,
                "max_input_chars": 10000
            }
        },
        "recomendaciones": [
            "Mantener conversaciones bajo 4 intercambios",
            "Usar 'generar auditoria' para reiniciar el historial",
            "Documentos muy largos dividirlos en partes"
        ]
    }

# --- Endpoint de salud ---
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "GLY-IA API is running"}

# --- Configuración del servidor ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=1,
        limit_concurrency=100,
        timeout_keep_alive=5
    )
