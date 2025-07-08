from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gly_ia import gly_ia

app = FastAPI()

class ConsultaInput(BaseModel):
    query: str
    rol: str = "Auditor"
    temperatura: float = 0.7
    estilo: str = "Formal"

@app.post("/gpt")
def procesar_consulta(data: ConsultaInput):
    try:
        respuesta = gly_ia(
            query=data.query,
            rol=data.rol,
            temperatura=data.temperatura,
            estilo=data.estilo
        )
        return {"respuesta": respuesta if isinstance(respuesta, str) else respuesta.get("text", "Sin respuesta.")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
