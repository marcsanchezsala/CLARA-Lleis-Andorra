"""
app.py
==============================================================================
Fase 4 — Interfície i Observabilitat (Chainlit) del Sistema RAG Legal
==============================================================================

Este script expone el grafo de LangGraph construido en la Fase 3
(`src/agent.py`) a través de una interfaz de chat con Chainlit, mostrando
en tiempo real:

  1. Los pasos internos del agente (Rewrite, Retrieve_and_Rerank,
     Evaluate_Relevance) como `cl.Step` colapsables.
  2. El streaming token a token de la respuesta final, separando el bloque
     de razonamiento `<think>...</think>` (que va a un Step aparte) de la
     respuesta limpia en catalán (que va al `cl.Message` principal).
  3. Los fragmentos legales (chunks) realmente utilizados, como
     `cl.Text` elements adjuntos al mensaje final (citas/fuentes del BOPA).

Ejecución:
    chainlit run app.py -w

------------------------------------------------------------------------------
SOBRE LA SINCRONIZACIÓN LangGraph (generador async) <-> Chainlit (UI async)
------------------------------------------------------------------------------
Chainlit corre sobre un único bucle de eventos asyncio (vía FastAPI/Starlette).
LangGraph expone `graph.astream_events(...)`, un GENERADOR ASÍNCRONO que va
"emitiendo" eventos (inicio/fin de nodo, tokens de LLM, etc.) a medida que el
grafo avanza. Consumimos ese generador con `async for event in ...:` dentro
del propio handler `async def on_message`, que Chainlit también ejecuta en
el mismo bucle de eventos.

Esto es clave: como TODO es `async` (los nodos de agent.py, el generador de
LangGraph, y los métodos de Chainlit como `step.send()`/`message.stream_token()`),
cada `await` cede el control al bucle de eventos sin bloquear ningún hilo.
Así, mientras el LLM está "pensando" o el retriever está buscando (que
internamente corre en un hilo aparte vía `asyncio.to_thread`, ver
`agent.py::retrieve_and_rerank_node`), Chainlit puede seguir sirviendo la UI,
refrescar el spinner de los Steps y aceptar nuevos eventos del cliente, todo
en el mismo proceso sin necesidad de threads ni colas manuales.

Autor: Ingeniería de Software - Interfaces de Usuario para IA / Sistemas RAG
==============================================================================
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import chainlit as cl

from agent import AgentState, build_graph
from retriever import LegalRetriever, ResultatFinal

# ==============================================================================
# 0. CONFIGURACIÓN DE OBSERVABILIDAD
# ==============================================================================

# Nombre visible (colapsable) de cada `cl.Step`, indexado por el nombre real
# del nodo tal como se registró en `build_graph()` (src/agent.py). Solo estos
# tres nodos generan un Step propio; `generate_with_thinking` se trata aparte
# porque necesita streaming token a token (ver más abajo).
NODE_STEP_CONFIG: Dict[str, Dict[str, str]] = {
    "rewrite": {"name": "🔍 Reescrivint la consulta...", "type": "tool"},
    "retrieve_and_rerank": {"name": "📚 Cercant en les lleis andorranes...", "type": "retrieval"},
    "evaluate_relevance": {"name": "✅ Avaluant la rellevància dels fragments...", "type": "tool"},
}

# Nombre del nodo final, cuyo streaming de tokens se intercepta manualmente.
GENERATE_NODE_NAME = "generate_with_thinking"

THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


# ==============================================================================
# 1. INICIALIZACIÓN DE SESIÓN (@cl.on_chat_start)
# ==============================================================================

@cl.on_chat_start
async def on_chat_start() -> None:
    """
    Se ejecuta una única vez por sesión/pestaña de usuario.

    Carga el LegalRetriever (pesado: BM25, ChromaDB, embeddings,
    Cross-Encoder) y construye el grafo de LangGraph UNA SOLA VEZ,
    guardando ambos en `cl.user_session`. Así, cada mensaje posterior del
    usuario reutiliza el mismo grafo/modelos ya cargados en memoria, en
    vez de recargarlos en cada turno de conversación.

    `LegalRetriever()` es una llamada síncrona y bloqueante (carga modelos
    de sentence-transformers/cross-encoder desde disco). La ejecutamos con
    `asyncio.to_thread` para no congelar el bucle de eventos de Chainlit
    mientras el usuario espera a que el sistema arranque.
    """
    aviso = cl.Message(content="⚙️ Carregant el sistema RAG legal (índexs i models)...")
    await aviso.send()

    retriever: LegalRetriever = await asyncio.to_thread(LegalRetriever)
    graph = build_graph(retriever=retriever)

    # Guardamos en la sesión del usuario para reutilizar en @cl.on_message.
    cl.user_session.set("graph", graph)
    cl.user_session.set("retriever", retriever)

    aviso.content = "✅ Sistema llest. Fes-me una pregunta sobre la legislació andorrana."
    await aviso.update()


# ==============================================================================
# 2. UTILIDAD: detección de tags <think> partidos entre chunks de streaming
# ==============================================================================

def _sufijo_parcial_de_tag(buffer: str, tag: str) -> str:
    """
    Dado un `buffer` de texto y un `tag` completo (p. ej. "<think>"),
    devuelve el sufijo más largo de `buffer` que coincide con un PREFIJO
    de `tag`. Sirve para "retener" en el buffer la posible mitad de una
    etiqueta que aún no ha llegado entera (los tokens del LLM no respetan
    los límites de las etiquetas XML, así que "<thi" y "nk>" pueden llegar
    en chunks separados).

    Ejemplo: buffer="...text <thi", tag="<think>" -> devuelve "<thi".

    Args:
        buffer: texto acumulado aún no emitido.
        tag: etiqueta completa a buscar ("<think>" o "</think>").

    Returns:
        El sufijo parcial coincidente, o "" si no hay ninguna coincidencia.
    """
    max_len = min(len(buffer), len(tag) - 1)
    for length in range(max_len, 0, -1):
        if buffer.endswith(tag[:length]):
            return buffer[-length:]
    return ""


# ==============================================================================
# 3. MANEJO DE MENSAJES (@cl.on_message)
# ==============================================================================

@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Procesa cada mensaje del usuario invocando el grafo de LangGraph y
    reflejando en la UI de Chainlit, en tiempo real:

      - Un `cl.Step` colapsable por cada nodo de observabilidad
        (rewrite, retrieve_and_rerank, evaluate_relevance).
      - El streaming del nodo final `generate_with_thinking`, separando
        `<think>...</think>` (-> Step "Raonament del model") de la
        respuesta limpia en catalán (-> `cl.Message` principal).
      - Los chunks legales realmente recuperados, como `cl.Text` elements
        adjuntos al mensaje final (citas/fuentes).
    """
    graph = cl.user_session.get("graph")
    if graph is None:
        await cl.Message(
            content="⚠️ El sistema encara s'està inicialitzant, torna-ho a provar en uns segons."
        ).send()
        return

    initial_state: AgentState = {
        # IMPORTANTE: en versiones recientes de Chainlit, `message.content`
        # no es un `str` plano sino un `TextAccessor` (proxy de acceso
        # perezoso al texto). Si lo metemos tal cual en el AgentState, ese
        # objeto viaja por todo el grafo y puede acabar filtrándose donde
        # se espera un `str`/`dict`. Lo convertimos explícitamente aquí,
        # en el único punto de entrada del texto del usuario al sistema.
        "query": str(message.content),
        "query_reescrita": "",
        "documentos": [],
        "respuesta_final": "",
    }

    # Steps de observabilidad actualmente abiertos, indexados por nombre de
    # nodo de LangGraph (evita abrir el mismo Step dos veces si el nodo
    # emite varios eventos "on_chain_start" internos).
    open_steps: Dict[str, cl.Step] = {}

    # Step colapsable donde se vuelca el contenido de <think>...</think>.
    # Se crea de forma perezosa, solo cuando detectamos la apertura del tag.
    thinking_step: Optional[cl.Step] = None

    # Mensaje principal del chat: SOLO recibirá la respuesta final limpia
    # (fuera del bloque <think>), vía streaming token a token.
    respuesta_msg = cl.Message(content="")
    await respuesta_msg.send()

    # Buffer de texto sin procesar todavía: puede contener el final de un
    # tag partido a la espera de más tokens (ver `_sufijo_parcial_de_tag`).
    buffer = ""
    dentro_de_think = False

    # Guardamos aquí los documentos reales devueltos por el nodo de
    # recuperación, para poder adjuntarlos como citas al mensaje final.
    documentos_recuperados: List[ResultatFinal] = []

    # --------------------------------------------------------------------
    # Consumo del generador asíncrono de LangGraph. `version="v2"` es el
    # esquema de eventos estable de LangChain/LangGraph que incluye tanto
    # eventos a nivel de nodo ("on_chain_start"/"on_chain_end") como
    # eventos a nivel de token de cualquier ChatModel invocado dentro de
    # un nodo ("on_chat_model_stream").
    #
    # ATENCIÓN — dos trampas importantes al filtrar estos eventos:
    #
    #   1. `event["metadata"]["langgraph_node"]` identifica de qué nodo
    #      "viene" un evento, pero LangGraph PROPAGA ese metadato a TODOS
    #      los runnables anidados dentro del nodo (el ChatOllama interno,
    #      el StrOutputParser, etc.), no solo al nodo en sí. Si filtramos
    #      solo por `langgraph_node`, capturamos también los
    #      "on_chain_end" de esas llamadas internas, cuyo `output` NO es
    #      el AgentState del nodo sino, p. ej., un string suelto o el
    #      objeto Pydantic de salida estructurada -> provoca errores tipo
    #      "'X' object has no attribute 'get'" al tratarlo como dict.
    #      Por eso añadimos `event["name"] == nombre_nodo`: LangGraph
    #      nombra el runnable de CADA NODO exactamente igual que su id
    #      ("rewrite", "retrieve_and_rerank", ...), mientras que las
    #      llamadas internas conservan su propio nombre de clase
    #      ("ChatOllama", "RunnableSequence", ...). Esta doble condición
    #      aísla el evento de nodo "de verdad".
    #
    #   2. Si el `async for` termina por una excepción, el generador
    #      asíncrono de `astream_events` puede quedar sin cerrar
    #      correctamente (Python lo recoge más tarde en el GC, generando
    #      el aviso "async generator ignored GeneratorExit" en los logs).
    #      Lo cerramos explícitamente en un `finally` con `aclose()`.
    # --------------------------------------------------------------------
    eventos_grafo = graph.astream_events(initial_state, version="v2")
    try:
        async for event in eventos_grafo:
            tipo_evento = event["event"]
            nombre_nodo = event.get("metadata", {}).get("langgraph_node")
            nombre_runnable = event.get("name")

            # --- A. Un nodo observable EMPIEZA -> abrimos su Step -------
            if (
                tipo_evento == "on_chain_start"
                and nombre_nodo in NODE_STEP_CONFIG
                and nombre_runnable == nombre_nodo
                and nombre_nodo not in open_steps
            ):
                config = NODE_STEP_CONFIG[nombre_nodo]
                step = cl.Step(name=config["name"], type=config["type"])
                await step.send()
                open_steps[nombre_nodo] = step

            # --- B. Un nodo observable TERMINA -> rellenamos y cerramos su Step
            elif (
                tipo_evento == "on_chain_end"
                and nombre_nodo in open_steps
                and nombre_runnable == nombre_nodo
            ):
                salida = event["data"].get("output")

                # Blindaje defensivo: si por lo que sea `salida` no es el
                # AgentState (dict) esperado, no reventamos la conversación
                # entera — simplemente ignoramos ese evento concreto y
                # seguimos esperando al evento de nodo correcto.
                if not isinstance(salida, dict):
                    continue

                step = open_steps.pop(nombre_nodo)

                if nombre_nodo == "rewrite":
                    query_reescrita = salida.get("query_reescrita", "")
                    step.output = f"**Query reescrita per a la cerca:**\n\n`{query_reescrita}`"

                elif nombre_nodo == "retrieve_and_rerank":
                    documentos_recuperados = salida.get("documentos", [])
                    if documentos_recuperados:
                        llistat = "\n".join(
                            f"- **{doc['titol_chunk']}** _(score={doc['score_cross_encoder']:.3f})_"
                            for doc in documentos_recuperados
                        )
                    else:
                        llistat = "_No s'ha trobat cap fragment rellevant._"
                    step.output = f"**Fragments recuperats ({len(documentos_recuperados)}):**\n\n{llistat}"

                elif nombre_nodo == "evaluate_relevance":
                    step.output = "Avaluació de rellevància dels fragments completada."

                await step.update()

            # --- C. Streaming token a token del NODO FINAL de generación ----
            elif tipo_evento == "on_chat_model_stream" and nombre_nodo == GENERATE_NODE_NAME:
                chunk_contenido = event["data"]["chunk"].content
                if not chunk_contenido:
                    continue

                buffer += chunk_contenido

                # Procesamos el buffer en bucle porque un único chunk puede
                # contener, por ejemplo, el cierre de </think> Y el inicio de
                # la respuesta final a la vez; hay que repartir ambas partes.
                while True:
                    if not dentro_de_think:
                        idx_apertura = buffer.find(THINK_OPEN_TAG)

                        if idx_apertura == -1:
                            # Sin tag de apertura completo todavía: puede que la
                            # cola del buffer sea el principio de "<think>", así
                            # que la retenemos y enviamos el resto como
                            # respuesta final limpia (streaming al cl.Message).
                            cola_pendiente = _sufijo_parcial_de_tag(buffer, THINK_OPEN_TAG)
                            texto_seguro = buffer[: len(buffer) - len(cola_pendiente)]
                            if texto_seguro:
                                await respuesta_msg.stream_token(texto_seguro)
                            buffer = cola_pendiente
                            break

                        # Todo lo anterior al tag de apertura es respuesta final.
                        texto_antes = buffer[:idx_apertura]
                        if texto_antes:
                            await respuesta_msg.stream_token(texto_antes)

                        buffer = buffer[idx_apertura + len(THINK_OPEN_TAG):]
                        dentro_de_think = True

                        # Abrimos (de forma perezosa) el Step de razonamiento.
                        thinking_step = cl.Step(name="🧠 Raonament del model", type="tool")
                        thinking_step.output = ""
                        await thinking_step.send()
                        # Continuamos el bucle: puede quedar más contenido en buffer.

                    else:
                        idx_cierre = buffer.find(THINK_CLOSE_TAG)

                        if idx_cierre == -1:
                            cola_pendiente = _sufijo_parcial_de_tag(buffer, THINK_CLOSE_TAG)
                            texto_seguro = buffer[: len(buffer) - len(cola_pendiente)]
                            if texto_seguro and thinking_step is not None:
                                thinking_step.output += texto_seguro
                                await thinking_step.update()
                            buffer = cola_pendiente
                            break

                        # Todo lo anterior al tag de cierre es razonamiento interno.
                        texto_dentro = buffer[:idx_cierre]
                        if texto_dentro and thinking_step is not None:
                            thinking_step.output += texto_dentro
                            await thinking_step.update()

                        buffer = buffer[idx_cierre + len(THINK_CLOSE_TAG):]
                        dentro_de_think = False
                        thinking_step = None
                        # Continuamos el bucle: el resto del buffer ya es
                        # respuesta final limpia (se procesará en la próxima
                        # iteración del `while True`, rama `not dentro_de_think`).
    finally:
        # Cierre explícito y ordenado del generador asíncrono de LangGraph,
        # tanto si el bucle terminó normalmente como si saltó una excepción.
        # Evita el aviso "async generator ignored GeneratorExit" en los logs.
        await eventos_grafo.aclose()

    # Al terminar el streaming, si queda algún resto seguro en el buffer
    # (y no estamos a medio tag), lo emitimos como cola de la respuesta.
    if buffer and not dentro_de_think:
        await respuesta_msg.stream_token(buffer)

    # --------------------------------------------------------------------
    # 4. CITAS: adjuntamos los fragmentos legales realmente usados como
    #    `cl.Text` elements del mensaje final, para que el usuario pueda
    #    consultar el texto original del BOPA sin salir del chat.
    # --------------------------------------------------------------------
    if documentos_recuperados:
        respuesta_msg.elements = [
            cl.Text(
                name=f"[{i}] {doc['titol_chunk']}",
                content=doc["text"],
                display="side",
            )
            for i, doc in enumerate(documentos_recuperados, start=1)
        ]

    await respuesta_msg.update()