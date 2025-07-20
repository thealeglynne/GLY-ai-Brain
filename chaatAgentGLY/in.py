import asyncio
from gly_ia import gly_ia, get_current_user, get_user_audits
from gly_dev import generar_documento_consultivo  # Asumiendo que esta función existe

async def main():
    rol = "Auditor"
    estilo = "Conversacional"
    temperatura = 0.7

    print("💬 Bienvenido a GLY-IA (interactivo). Escribe 'salir' para terminar.\n")

    while True:
        entrada = input("👤 Tú: ")
        if entrada.strip().lower() == "salir":
            print("👋 Saliendo.\n")

            # Obtener el usuario autenticado y sus auditorías
            user = await get_current_user()
            if not user:
                print("❌ No se pudo obtener el usuario autenticado. No se generará la propuesta.")
                break

            audits = await get_user_audits(user.id)
            if not audits:
                print("❌ No se encontraron auditorías previas para generar la propuesta.")
                break

            print("\n🛠️ Generando propuesta técnica basada en las auditorías previas...\n")
            propuesta = generar_documento_consultivo(audits)  # Pasar las auditorías de Supabase

            # Guardar propuesta en archivo
            with open("propuesta_tecnica_glynne.txt", "w", encoding="utf-8") as f:
                f.write(propuesta)

            print("✅ Propuesta técnica generada y guardada en 'propuesta_tecnica_glynne.txt'\n")
            print("📄 Resumen:\n")
            print(propuesta[:1000] + "\n...")
            break

        respuesta, audits = await gly_ia(
            entrada,
            rol=rol,
            estilo=estilo,
            temperatura=temperatura
        )
        print(f"🤖 GLY-IA: {respuesta}\n")

if __name__ == "__main__":
    asyncio.run(main())