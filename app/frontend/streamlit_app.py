import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="AI Buscador RAG", layout="wide")
st.title("AI Buscador RAG local")


def api_get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, **kwargs):
    response = requests.post(f"{API_BASE_URL}{path}", timeout=180, **kwargs)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{response.status_code} {detail}")
    return response.json()


def api_delete(path: str, **kwargs):
    response = requests.delete(f"{API_BASE_URL}{path}", timeout=60, **kwargs)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{response.status_code} {detail}")
    return response.json()


def refresh_documents():
    try:
        workspace_id = st.session_state.get("workspace_id")
        path = f"/documents?workspace_id={workspace_id}" if workspace_id else "/documents"
        st.session_state["documents"] = api_get(path)
    except Exception as exc:
        st.error(f"No se pudieron cargar documentos: {exc}")


def refresh_workspaces():
    try:
        workspaces = api_get("/workspaces")
        st.session_state["workspaces"] = workspaces
        if workspaces and not st.session_state.get("workspace_id"):
            st.session_state["workspace_id"] = workspaces[0]["id"]
    except Exception as exc:
        st.error(f"No se pudieron cargar workspaces: {exc}")


if "workspaces" not in st.session_state:
    refresh_workspaces()


with st.sidebar:
    st.header("Workspace")
    workspaces = st.session_state.get("workspaces", [])
    if workspaces:
        workspace_names = {f"{workspace['name']} (ID {workspace['id']})": workspace["id"] for workspace in workspaces}
        current_id = st.session_state.get("workspace_id", workspaces[0]["id"])
        selected_label = st.selectbox(
            "Alcance activo",
            options=list(workspace_names.keys()),
            index=list(workspace_names.values()).index(current_id)
            if current_id in workspace_names.values()
            else 0,
        )
        selected_workspace_id = workspace_names[selected_label]
        if selected_workspace_id != st.session_state.get("workspace_id"):
            st.session_state["workspace_id"] = selected_workspace_id
            refresh_documents()
            st.rerun()
    else:
        selected_workspace_id = None
        st.warning("Creá un workspace para cargar información.")

    new_workspace = st.text_input("Nuevo workspace")
    if st.button("Crear workspace", disabled=not new_workspace.strip()):
        try:
            workspace = api_post("/workspaces", json={"name": new_workspace.strip()})
            st.session_state["workspace_id"] = workspace["id"]
            refresh_workspaces()
            refresh_documents()
            st.rerun()
        except Exception as exc:
            st.error(f"No se pudo crear el workspace: {exc}")

    st.divider()
    st.header("Carga")
    uploaded_files = st.file_uploader(
        "Archivos",
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls", "cs", "sql"],
        accept_multiple_files=True,
    )
    if st.button("Cargar archivos", disabled=not uploaded_files or selected_workspace_id is None):
        try:
            loaded = 0
            for uploaded_file in uploaded_files:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                result = api_post("/documents/upload", data={"workspace_id": selected_workspace_id}, files=files)
                loaded += 1
            st.success(f"Archivos cargados: {loaded}")
            refresh_documents()
        except Exception as exc:
            st.error(f"Error al cargar archivo: {exc}")

    st.divider()
    folder_path = st.text_input("Carpeta local recursiva", placeholder=r"D:\datos\documentos")
    if st.button("Cargar carpeta", disabled=not folder_path.strip() or selected_workspace_id is None):
        try:
            result = api_post(
                "/documents/folder",
                json={"workspace_id": selected_workspace_id, "folder_path": folder_path.strip()},
            )
            st.success(f"Archivos cargados: {len(result['loaded'])} de {result['total_found']}")
            if result["errors"]:
                with st.expander("Errores de carga"):
                    st.json(result["errors"])
            refresh_documents()
        except Exception as exc:
            st.error(f"Error al cargar carpeta: {exc}")

    st.divider()
    text_name = st.text_input("Nombre del texto", value="Texto manual")
    manual_text = st.text_area("Texto manual", height=180)
    if st.button("Guardar texto", disabled=not manual_text.strip() or selected_workspace_id is None):
        try:
            result = api_post(
                "/documents/text",
                json={"workspace_id": selected_workspace_id, "name": text_name, "text": manual_text},
            )
            st.success(f"Texto cargado: {result['chunks']} chunks")
            refresh_documents()
        except Exception as exc:
            st.error(f"Error al cargar texto: {exc}")


left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("Documentos cargados")
    if st.button("Actualizar lista"):
        refresh_documents()
    if st.button("Eliminar repetidos", disabled=st.session_state.get("workspace_id") is None):
        try:
            result = api_delete(
                "/documents/duplicates",
                params={"workspace_id": st.session_state.get("workspace_id")},
            )
            st.success(f"Documentos repetidos eliminados: {result['deleted_count']}")
            refresh_documents()
            st.rerun()
        except Exception as exc:
            st.error(f"No se pudieron eliminar repetidos: {exc}")
    if "documents" not in st.session_state:
        refresh_documents()

    documents = st.session_state.get("documents", [])
    if not documents:
        st.info("Todavía no hay documentos cargados.")
    for doc in documents:
        with st.container(border=True):
            st.markdown(f"**{doc['name']}**")
            st.caption(f"ID {doc['id']} | {doc['workspace_name']} | {doc['file_type']} | {doc['chunks']} chunks")
            st.caption(doc["created_at"])
            if st.button("Eliminar", key=f"delete-{doc['id']}"):
                try:
                    api_delete(f"/documents/{doc['id']}")
                    refresh_documents()
                    st.rerun()
                except Exception as exc:
                    st.error(f"No se pudo eliminar: {exc}")


with right:
    st.subheader("Consulta")
    question = st.text_area(
        "Pregunta",
        placeholder="Ej: Qué conclusiones se pueden sacar? Respondé solo usando la información cargada.",
        height=120,
    )
    top_k = st.slider("Chunks a recuperar", min_value=1, max_value=20, value=5)

    if st.button(
        "Consultar",
        type="primary",
        disabled=not question.strip() or st.session_state.get("workspace_id") is None,
    ):
        try:
            result = api_post(
                "/chat/query",
                json={"workspace_id": st.session_state.get("workspace_id"), "question": question, "top_k": top_k},
            )
            st.markdown("### Respuesta")
            st.write(result["answer"])
            st.caption(f"Tokens de contexto enviados al prompt: {result['context_tokens']}")

            st.markdown("### Fuentes usadas")
            if not result["sources"]:
                st.info("No se usaron fuentes.")
            for source in result["sources"]:
                with st.expander(f"{source['document_name']} | chunk {source['chunk_index']}"):
                    distance = source.get("similarity_distance")
                    lexical_score = source.get("lexical_score")
                    metrics = [f"Documento ID: {source['document_id']}"]
                    if distance is not None:
                        metrics.append(f"distancia: {distance:.4f}")
                    if lexical_score is not None:
                        metrics.append(f"coincidencia literal: {lexical_score}")
                    st.caption(" | ".join(metrics))
                    st.write(source["preview"])
        except Exception as exc:
            st.error(f"Error en consulta: {exc}")
