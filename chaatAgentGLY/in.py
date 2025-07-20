from gly_ia import gly_ia

from gly_dev import generar_documento_consultivo

historial = []
rol = "Auditor"
estilo = "Formal"
temperatura = 0.7

print("💬 Bienvenido a GLY-IA (interactivo). Escribe 'salir' para terminar.\n")

while True:
    entrada = input("👤 Tú: ")
    if entrada.strip().lower() == "salir":
        print("👋 Saliendo. La conversación se ha guardado en 'conversacion_gly_ia.json'")
        
        print("\n🛠️ Generando propuesta técnica basada en esta conversación...\n")
        propuesta = generar_documento_consultivo()

        # Guardar propuesta en archivo
        with open("propuesta_tecnica_glynne.txt", "w", encoding="utf-8") as f:
            f.write(propuesta)

        print("✅ Propuesta técnica generada y guardada en 'propuesta_tecnica_glynne.txt'\n")
        print("📄 Resumen:\n")
        print(propuesta[:1000] + "\n...")  # Muestra solo un extracto

        break

    respuesta, historial = gly_ia(
        entrada,
        rol=rol,
        estilo=estilo,
        temperatura=temperatura,
        historial=historial
    )
    print(f"🤖 GLY-IA: {respuesta}\n")
