# AI Buscador RAG local

Aplicación local tipo RAG para cargar documentos, generar embeddings locales, guardarlos en ChromaDB y consultar la información con un LLM configurable.

## Funcionalidades

- Carga de texto manual, PDF, DOCX, TXT, CSV, XLSX, XLS, CS y SQL.
- Workspaces para delimitar el alcance de documentos, embeddings y consultas.
- Carga recursiva de carpetas locales desde el backend.
- Detección de documentos repetidos por contenido dentro de cada workspace.
- Extracción y limpieza básica de texto.
- División en chunks con overlap configurable.
- Embeddings locales con `sentence-transformers`.
- Base vectorial local con ChromaDB.
- Metadatos en SQLite.
- Consulta semántica y envío al LLM solo del contexto relevante.
- Fuentes/chunks visibles en cada respuesta.
- Proveedores LLM: OpenAI, Groq, Ollama y LM Studio.

## Instalación

Requiere Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r app/requirements.txt
```

Si ya habías instalado dependencias antes de este cambio, actualizá `posthog` para evitar errores de telemetría de ChromaDB:

```bash
pip install "posthog<4.0.0"
```

En Linux/macOS:

```bash
source .venv/bin/activate
```

## Configuración

Crear `app/.env` si querés cambiar valores:

```env
LLM_PROVIDER=ollama
MODEL_NAME=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=
GROQ_API_KEY=gsk_
```

Valores válidos para `LLM_PROVIDER`:

- `ollama`
- `lmstudio`
- `openai`
- `groq`

Para Ollama:

```bash
ollama pull llama3.1
ollama serve
```

## Ejecución

Backend FastAPI:

```bash
uvicorn app.main:app --reload
```

Frontend Streamlit, en otra terminal:

```bash
streamlit run app/frontend/streamlit_app.py
```

Abrir:

- API: `http://127.0.0.1:8000/docs`
- Frontend: `http://localhost:8501`

## Endpoints

- `POST /workspaces`
- `GET /workspaces`
- `DELETE /workspaces/{id}`
- `POST /documents/upload`
- `POST /documents/folder`
- `POST /documents/text`
- `GET /documents`
- `DELETE /documents/duplicates`
- `POST /chat/query`
- `DELETE /documents/{id}`

## Ejemplo de uso

1. Iniciar backend y frontend.
2. Crear o seleccionar un workspace.
3. Cargar `app/sample_data/demo.txt` desde la interfaz, pegar texto manual o indicar una carpeta local para carga recursiva.
4. Consultar:

```text
Qué temas aparecen más repetidos?
```

```text
Qué conclusiones se pueden sacar?
```

```text
Qué datos faltan?
```

La respuesta debe basarse solo en los chunks recuperados. Si el contexto no alcanza, el sistema instruye al LLM a responder:

```text
No hay información suficiente en los documentos cargados.
```

## Embeddings vs tokens

- Los embeddings se usan para búsqueda semántica en ChromaDB.
- Cada embedding queda asociado al workspace del documento cargado.
- Los tokens se usan para estimar y controlar cuánto contexto se envía al prompt del LLM.
- El parámetro `max_context_tokens` evita enviar documentos completos cuando solo se necesitan fragmentos relevantes.

## Workspaces y alcance

Cada documento pertenece a un workspace. Al consultar, la búsqueda semántica se filtra por ese workspace, por lo que el LLM solo recibe chunks de ese alcance.

Los duplicados se detectan por hash del contenido normalizado dentro del mismo workspace. Si intentás cargar el mismo contenido dos veces, la app lo rechaza. Desde la interfaz también podés usar `Eliminar repetidos` para borrar duplicados antiguos y conservar el primer documento cargado.

La app crea automáticamente un workspace `Default` al iniciar. Para cargar una carpeta completa desde la API:

```bash
curl -X POST http://127.0.0.1:8000/documents/folder \
  -H "Content-Type: application/json" \
  -d "{\"workspace_id\":1,\"folder_path\":\"D:\\\\datos\\\\documentos\"}"
```

La carga de carpeta recorre subcarpetas y procesa solo extensiones soportadas: PDF, DOCX, TXT, CSV, XLSX, XLS, CS y SQL.

## Datos locales

La app crea automáticamente:

- `storage/rag.db`: SQLite con documentos y chunks.
- `storage/uploads/`: archivos cargados.
- `storage/chroma/`: índice vectorial persistente de ChromaDB.

## Notas

La primera carga puede tardar porque `sentence-transformers` descarga el modelo de embeddings. Si no hay internet, descargá previamente el modelo o configurá `embedding_model` con una ruta local.

Si `/chat/query` devuelve error, revisá que el proveedor LLM configurado en `.env` esté disponible. Para `LLM_PROVIDER=ollama`, Ollama debe estar corriendo y el modelo de `MODEL_NAME` debe estar descargado.
