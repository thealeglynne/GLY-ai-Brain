from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chaatAgentGLY import gly_ia
import os

app = FastAPI()

# --- Configuración de CORS ---
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

# --- Modelo de entrada ---
class ConsultaInput(BaseModel):
    query: str
    rol: str = "Auditor"
    temperatura: float = 0.7
    estilo: str = "Formal"

# --- Endpoint principal ---
@app.post("/gpt")
async def procesar_consulta(data: ConsultaInput):
    try:
        respuesta = gly_ia(
            query=data.query,
            rol=data.rol,
            temperatura=data.temperatura,
            estilo=data.estilo
        )

        # Aseguramos respuesta tipo string
        texto_respuesta = respuesta if isinstance(respuesta, str) else str(respuesta)
        return {"respuesta": texto_respuesta}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# --- Health Check ---
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "GLY-IA API is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)