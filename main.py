from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Importar funciones ---
try:
    from chaatAgentGLY.gly_ia import gly_ia
    from chaatAgentGLY.gly_dev import generar_documento_consultivo
except ImportError as e:
    logger.error(f"Error al importar agentes: {e}")
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

# --- Entrada esperada desde el frontend ---
class ConsultaInput(BaseModel):
    query: str
    user_id: str  # ¡Este campo es obligatorio ahora!
    rol: str = "Auditor"
    temperatura: float = 0.7
    estilo: str = "Formal"

# --- Endpoint principal del agente GLY-IA ---
@app.post("/gpt")
async def procesar_consulta(data: ConsultaInput):
    try:
        if not data.query.strip():
            raise HTTPException(status_code=400, detail="El campo 'query' no puede estar vacío.")

        if not data.user_id.strip():
            raise HTTPException(status_code=400, detail="El campo 'user_id' es obligatorio.")

        # Detectar trigger para generar propuesta técnica
        if data.query.strip().lower() == "generar auditoria":
            logger.info(f"Generando auditoría para usuario {data.user_id}...")
            propuesta = generar_documento_consultivo(data.user_id)
            logger.info("Auditoría generada exitosamente")
            return {
                "respuesta": "Auditoría finalizada. Propuesta técnica generada.",
                "propuesta": propuesta
            }

        # Caso normal: continuar conversación con GLY-IA
        respuesta, _ = await gly_ia(
            query=data.query,
            user_id=data.user_id,
            rol=data.rol,
            estilo=data.estilo,
            temperatura=data.temperatura
        )

        return {"respuesta": respuesta}

    except Exception as e:
        logger.error(f"Error interno en /gpt: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# --- Endpoint adicional para generar propuesta técnica manual ---
@app.get("/propuesta-tecnica/{user_id}")
async def generar_propuesta(user_id: str):
    try:
        if not user_id.strip():
            raise HTTPException(status_code=400, detail="El campo 'user_id' es obligatorio.")
        propuesta = generar_documento_consultivo(user_id)
        return {"propuesta": propuesta}
    except Exception as e:
        logger.error(f"Error al generar propuesta: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al generar propuesta: {str(e)}")

# --- Endpoint de salud ---
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "GLY-IA API is running"}

# --- Ejecución local ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
