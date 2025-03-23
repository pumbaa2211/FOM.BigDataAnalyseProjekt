# Demonstriert den Einsatz von Umgebungsvariablen.
# Es muss eine Datei namens ".env" im Projektordner erstellt werden, die den OpenAI API-Schlüssel enthält.

import os
from dotenv import load_dotenv  # Füge diesen Import hinzu

load_dotenv()

openai_key = os.getenv("OPENAI_KEY")
print(f"OpenAI API Key: {openai_key}")