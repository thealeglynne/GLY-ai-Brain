from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Optional

# --- Configuración inicial ---
app = FastAPI()

# CORS más permisivo para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, reemplazar con tus dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic ---
class ConsultaInput(BaseModel):
    query: str
    rol: Optional[str] = "Auditor"
    temperatura: Optional[float] = 0.7
    estilo: Optional[str] = "Formal"

# --- Estado Global ---
historial_global = []

# --- Endpoints ---
@app.post("/gpt")
async def procesar_consulta(data: ConsultaInput, request: Request):
    try:
        # Debug: Verificar los datos recibidos
        print(f"Datos recibidos: {await request.json()}")
        
        if not data.query.strip():
            raise HTTPException(status_code=400, detail="El campo 'query' no puede estar vacío.")

        global historial_global

        # Manejar el caso de generar auditoría
        if data.query.strip().lower() == "generar auditoria":
            propuesta = generar_propuesta_tecnica()
            return {
                "respuesta": "✅ Auditoría finalizada. Propuesta técnica generada.",
                "propuesta": propuesta
            }

        # Procesar con GLY-IA
        respuesta, historial_global = gly_ia(
            data.query,
            rol=data.rol,
            estilo=data.estilo,
            temperatura=data.temperatura,
            historial=historial_global
        )

        return {"respuesta": respuesta}

    except Exception as e:
        print(f"Error en /gpt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoints adicionales ---
@app.get("/propuesta-tecnica")
async def generar_propuesta():
    try:
        propuesta = generar_documento_consultivo()
        return {"propuesta": propuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar propuesta: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "GLY-IA API is running"}

# --- Ejecución ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)