from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# --- Intentar importar la función real ---
try:
    from chaatAgentGLY import gly_ia
    if not callable(gly_ia):
        raise AttributeError("'gly_ia' no es una función")
except (ImportError, AttributeError):
    def gly_ia(query: str, rol: str, temperatura: float, estilo: str):
        return f"Respuesta simulada para: '{query}' con rol '{rol}', temperatura {temperatura}, estilo {estilo}."

# --- Inicializar la app ---
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
    if not data.query.strip():
        raise HTTPException(status_code=400, detail="El campo 'query' no puede estar vacío.")

    try:
        respuesta = gly_ia(
            query=data.query,
            rol=data.rol,
            temperatura=data.temperatura,
            estilo=data.estilo
        )
        return {"respuesta": str(respuesta)}

    except Exception as e:
        print("ERROR INTERNO:", str(e))
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

# --- Health Check ---
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "GLY-IA API is running"}

# --- Ejecutar en local ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
