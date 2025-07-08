from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chaatAgentGLY import gly_ia  # Asegúrate que esta función exista y esté importada correctamente

app = FastAPI()

# --- Configuración de CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gly-ai-arq.vercel.app/"],  # Cambia "*" por tu dominio real en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelo de entrada ---
class ConsultaInput(BaseModel):
    query: str
    rol: str = "Auditor"
    temperatura: float = 0.7
    estilo: str = "Formal"

# --- Endpoint principal ---
@app.post("/gpt")
def procesar_consulta(data: ConsultaInput):
    try:
        respuesta = gly_ia(
            query=data.query,
            rol=data.rol,
            temperatura=data.temperatura,
            estilo=data.estilo
        )

        # Aseguramos respuesta tipo string
        texto_respuesta = respuesta if isinstance(respuesta, str) else respuesta.get("text", "Sin respuesta.")
        return {"respuesta": texto_respuesta}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
