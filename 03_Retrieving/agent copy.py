"""
src/agent.py
==============================================================================
Fase 3 — El Grafo del Agente (Sistema RAG Agéntico para Textos Legales)
==============================================================================

Este módulo implementa el "cerebro" del agente legal usando LangGraph y un
modelo local Qwen2.5:7b servido a través de Ollama (ChatOllama).

Flujo del grafo:

    REWRITE -> RETRIEVE_AND_RERANK -> EVALUATE_RELEVANCE -> GENERATE_WITH_THINKING

    
>>> ollama pull qwen2.5:7b
    
Autor: Ingeniero de Datos - Sistemas RAG Agénticos
==============================================================================
"""

from __future__ import annotations

from typing import List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# Import de la clase REAL de la Fase 2 (src/retriever.py).
# Se soporta tanto la ejecución como paquete ("python -m src.agent" desde la
# raíz del proyecto) como la ejecución directa del script dentro de src/.
try:
    from src.retriever import LegalRetriever, ResultatFinal
except ImportError:  # pragma: no cover - fallback para ejecución directa
    from retriever import LegalRetriever, ResultatFinal

# ==============================================================================
# 0. CONFIGURACIÓN DEL MODELO
# ==============================================================================

# Modelo local Qwen2.5:7b servido vía Ollama.
# temperature=0.0 para las tareas deterministas (rewrite, evaluación).
LLM_REWRITE = ChatOllama(model="qwen2.5:7b", temperature=0.0)
LLM_EVALUATE = ChatOllama(model="qwen2.5:7b", temperature=0.0)
# Para la generación final se permite algo más de flexibilidad de redacción.
LLM_GENERATE = ChatOllama(model="qwen2.5:7b", temperature=0.2)


# ==============================================================================
# 1. ESTRUCTURA DEL ESTADO (STATE)
# ==============================================================================

class AgentState(TypedDict):
    """
    Estado compartido que fluye entre todos los nodos del grafo.

    Attributes:
        query: Pregunta original formulada por el usuario (lenguaje natural).
        query_reescrita: Consulta transformada en palabras clave óptimas
            para la búsqueda híbrida (semántica + léxica).
        documentos: Fragmentos legales recuperados y reordenados (rerank)
            por LegalRetriever.cercar(). Cada elemento es un ResultatFinal
            real (chunk_id, parent_id, tipus, titol_chunk, text,
            score_cross_encoder), NO un simple string.
        respuesta_final: Respuesta definitiva generada por el LLM, redactada
            en catalán y con el bloque de razonamiento <think>.
    """
    query: str
    query_reescrita: str
    documentos: List[ResultatFinal]
    respuesta_final: str


def _formatear_documentos(documentos: List[ResultatFinal]) -> str:
    """
    Convierte la lista de ResultatFinal recuperados por LegalRetriever en un
    bloque de texto legible por el LLM, conservando el título/artículo de
    cada chunk para que el modelo pueda citarlo explícitamente.

    Args:
        documentos: fragmentos legales devueltos por `retriever.cercar()`.

    Returns:
        Texto formateado, un fragmento por bloque, o un aviso si la lista
        está vacía.
    """
    if not documentos:
        return "(no s'ha recuperat cap fragment legal)"

    bloques = []
    for doc in documentos:
        bloques.append(
            f"[{doc['titol_chunk']}] (tipus: {doc['tipus']}, "
            f"score={doc['score_cross_encoder']:.4f})\n{doc['text']}"
        )
    return "\n---\n".join(bloques)


# ==============================================================================
# 2. NODO 1 — REWRITE (Reescritura de la consulta)
# ==============================================================================

_REWRITE_SYSTEM_PROMPT = """Ets un expert en dret laboral i documentació jurídica \
del BOPA (Butlletí Oficial del Principat d'Andorra). La teva única tasca és \
transformar la pregunta d'un usuari en una seqüència òptima de paraules clau \
per fer una cerca semàntica i lèxica (BM25) en una base de dades de textos legals.

Instruccions estrictes:
- Retorna EXCLUSIVAMENT la seqüència de paraules clau, sense cap explicació addicional.
- Inclou termes jurídics tècnics rellevants (lleis, articles, conceptes legals).
- No facis preguntes ni afegeixis puntuació innecessària.
- Escriu les paraules clau en català.

Exemple:
Pregunta: "¿Cuándo me pueden despedir?"
Paraules clau: Llei de relacions laborals despil·lament causes d'extinció del contracte
"""


def rewrite_node(state: AgentState) -> AgentState:
    """
    Nodo 1: Reescribe la consulta original del usuario en una secuencia de
    palabras clave optimizada para la recuperación híbrida (semántica + léxica).

    Args:
        state: Estado actual del agente (debe contener 'query').

    Returns:
        Estado actualizado con 'query_reescrita' rellenada.
    """
    messages = [
        SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=f'Pregunta: "{state["query"]}"\nParaules clau:'),
    ]

    chain = LLM_REWRITE | StrOutputParser()
    query_reescrita = chain.invoke(messages).strip()

    return {**state, "query_reescrita": query_reescrita}


# ==============================================================================
# 3. NODO 2 — RETRIEVE_AND_RERANK (Recuperación híbrida, Fase 2)
# ==============================================================================

def retrieve_and_rerank_node(state: AgentState, retriever: LegalRetriever) -> AgentState:
    """
    Nodo 2: Ejecuta la recuperación híbrida (Fase 2) delegando en la clase
    LegalRetriever real (src/retriever.py), que internamente combina:
        1. Búsqueda léxica BM25
        2. Búsqueda semántica ChromaDB
        3. Fusión Reciprocal Rank Fusion (RRF)
        4. Reranking con Cross-Encoder (BAAI/bge-reranker-v2-m3)

    Args:
        state: Estado actual del agente (debe contener 'query_reescrita').
        retriever: Instancia YA INICIALIZADA de LegalRetriever (inyectada
            mediante closure al construir el grafo — ver `build_graph`).
            La inicialización es costosa (carga modelos e índices), por eso
            debe hacerse una única vez fuera del grafo, nunca dentro del nodo.

    Returns:
        Estado actualizado con 'documentos' (fragmentos ganadores tras
        RRF + reranking, método público real `.cercar()`).
    """
    fragmentos_ganadores: List[ResultatFinal] = retriever.cercar(state["query_reescrita"])

    return {**state, "documentos": fragmentos_ganadores}


# ==============================================================================
# 4. NODO 3 — EVALUATE_RELEVANCE (Evaluación de relevancia con salida estructurada)
# ==============================================================================

class RelevanceEvaluation(BaseModel):
    """Esquema Pydantic que fuerza al LLM a devolver un JSON estructurado."""

    is_relevant: bool = Field(
        description=(
            "True si los fragmentos recuperados contienen información "
            "suficiente y pertinente para responder a la consulta del "
            "usuario. False en caso contrario."
        )
    )
    justificacion: str = Field(
        description="Breve justificación (1-2 frases) de la decisión tomada."
    )


_EVALUATE_SYSTEM_PROMPT = """Ets un avaluador jurídic estricte. Se't donarà la \
pregunta original d'un usuari i un conjunt de fragments de text legal recuperats \
d'una base de dades. La teva tasca és determinar si aquests fragments contenen \
informació SUFICIENT i RELLEVANT per respondre correctament la pregunta.

Sigues rigorós: si els fragments només toquen el tema tangencialment o no \
contenen cap article/normativa aplicable, marca'ls com a NO rellevants.
"""


def evaluate_relevance_node(state: AgentState) -> AgentState:
    """
    Nodo 3: Evalúa si los documentos recuperados son suficientemente
    relevantes para responder a la consulta original, forzando al modelo
    a devolver un JSON estructurado (is_relevant: bool).

    NOTA SOBRE EL FLUJO CONDICIONAL:
        - Si `is_relevant == True`  -> el grafo avanza al nodo
          `generate_with_thinking` (comportamiento actual, ver `build_graph`).
        - Si `is_relevant == False` -> AQUÍ es donde debería engancharse un
          nodo de fallback, por ejemplo:
            * `reformulate_query_node`: volver a REWRITE con una estrategia
              distinta (p. ej. ampliar sinónimos, relajar filtros de rerank).
            * `expand_retrieval_node`: aumentar `top_k` o bajar el umbral
              de similitud en el retriever.
            * `human_in_the_loop_node`: pedir aclaración al usuario.
          Esta rama condicional NO está implementada en este script (se deja
          preparada vía `route_after_evaluation`) para mantener el flujo
          lineal solicitado en los requisitos, pero la estructura del grafo
          ya admite añadirla con `graph.add_conditional_edges(...)`.

    Args:
        state: Estado actual del agente (debe contener 'query' y 'documentos').

    Returns:
        Estado sin modificar (la evaluación en sí no se persiste en el
        AgentState definido, pero se deja registrada aquí como comentario
        de diseño; si se desea persistirla, añadir el campo
        `es_relevante: bool` al TypedDict `AgentState`).
    """
    # Se fuerza la salida estructurada mediante el esquema Pydantic.
    structured_llm = LLM_EVALUATE.with_structured_output(RelevanceEvaluation)

    documentos_concatenados = _formatear_documentos(state["documentos"])

    messages = [
        SystemMessage(content=_EVALUATE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Pregunta original: {state['query']}\n\n"
                f"Fragments recuperats:\n{documentos_concatenados}\n\n"
                "Avalua la rellevància d'aquests fragments respecte a la pregunta."
            )
        ),
    ]

    evaluation: RelevanceEvaluation = structured_llm.invoke(messages)

    # Log interno de depuración (no forma parte del AgentState oficial).
    print(
        f"[evaluate_relevance_node] is_relevant={evaluation.is_relevant} "
        f"| justificacion={evaluation.justificacion}"
    )

    # --- BLOQUE CONDICIONAL (preparado, no activo en el flujo lineal) -------
    if not evaluation.is_relevant:
        # TODO(fallback): en un flujo no lineal, aquí se debería redirigir
        # el grafo hacia un nodo de reformulación de búsqueda o de
        # ampliación del retriever, en lugar de continuar hacia
        # generate_with_thinking. Ver `route_after_evaluation` más abajo.
        pass
    # --------------------------------------------------------------------

    return state


def route_after_evaluation(state: AgentState) -> str:
    """
    Función de enrutamiento condicional (opcional, lista para usar).

    Actualmente el grafo principal (`build_graph`) usa una arista fija
    Evaluate_Relevance -> Generate_With_Thinking, tal como piden los
    requisitos. Esta función queda preparada para quien quiera activar
    el enrutamiento condicional real, por ejemplo:

        graph.add_conditional_edges(
            "evaluate_relevance",
            route_after_evaluation,
            {
                "generar": "generate_with_thinking",
                "reformular": "rewrite",  # fallback: vuelve a reescribir
            },
        )

    Requeriría añadir `es_relevante: bool` al AgentState para poder leerlo
    aquí en lugar de recalcularlo.
    """
    # Placeholder: por defecto siempre continúa hacia la generación.
    return "generar"


# ==============================================================================
# 5. NODO 4 — GENERATE_WITH_THINKING (Generación razonada final)
# ==============================================================================

_GENERATE_SYSTEM_PROMPT = """Ets un assistent jurídic expert en normativa \
andorrana (BOPA). Has de respondre exclusivament basant-te en els fragments \
legals proporcionats.

ABANS de redactar la resposta final, HAS D'omplir OBLIGATÒRIAMENT el següent \
bloc de raonament, respectant EXACTAMENT aquest format XML:

<think>
1. Hechos planteados: ...
2. Artículos aplicables del BOPA: ...
3. Razonamiento deductivo: ...
</think>
[La teva resposta final aquí. Ha d'estar redactada EXCLUSIVAMENT en català, \
de forma clara, professional i citant els articles específics utilitzats.]

Normes estrictes:
- No t'inventis articles ni normativa que no aparegui als fragments proporcionats.
- Si els fragments no contenen prou informació, indica-ho clarament dins la resposta final.
- Mai ometis el bloc <think>...</think>.
- La resposta final (fora del bloc <think>) ha d'estar sempre en català.
"""


def generate_with_thinking_node(state: AgentState) -> AgentState:
    """
    Nodo 4: Genera la respuesta final razonada, obligando al modelo a
    completar un bloque <think> estructurado antes de la respuesta en catalán.

    Args:
        state: Estado actual del agente (debe contener 'query' y 'documentos').

    Returns:
        Estado actualizado con 'respuesta_final' rellenada (incluye el
        bloque <think> seguido de la respuesta en catalán).
    """
    documentos_concatenados = _formatear_documentos(state["documentos"])

    messages = [
        SystemMessage(content=_GENERATE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Pregunta de l'usuari: {state['query']}\n\n"
                f"Fragments legals recuperats:\n{documentos_concatenados}"
            )
        ),
    ]

    chain = LLM_GENERATE | StrOutputParser()
    respuesta_final = chain.invoke(messages).strip()

    return {**state, "respuesta_final": respuesta_final}


# ==============================================================================
# 6. CONSTRUCCIÓN DEL GRAFO
# ==============================================================================

def build_graph(retriever: "LegalRetriever"):
    """
    Construye y compila el StateGraph del agente legal.

    Flujo lineal:
        rewrite -> retrieve_and_rerank -> evaluate_relevance -> generate_with_thinking -> END

    Args:
        retriever: Instancia REAL de LegalRetriever (Fase 2, src/retriever.py)
            ya inicializada, que se inyecta en el nodo de recuperación
            mediante un closure, ya que los nodos de LangGraph solo reciben
            `state` como argumento posicional.

    Returns:
        Grafo compilado (CompiledGraph), listo para invocar con `.invoke(...)`.
    """
    graph = StateGraph(AgentState)

    # --- Registro de nodos ---------------------------------------------------
    graph.add_node("rewrite", rewrite_node)

    # retrieve_and_rerank_node necesita el retriever -> lo envolvemos en closure
    def _retrieve_and_rerank_with_retriever(state: AgentState) -> AgentState:
        return retrieve_and_rerank_node(state, retriever)

    graph.add_node("retrieve_and_rerank", _retrieve_and_rerank_with_retriever)
    graph.add_node("evaluate_relevance", evaluate_relevance_node)
    graph.add_node("generate_with_thinking", generate_with_thinking_node)

    # --- Definición de aristas (flujo lineal solicitado) ---------------------
    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve_and_rerank")
    graph.add_edge("retrieve_and_rerank", "evaluate_relevance")
    graph.add_edge("evaluate_relevance", "generate_with_thinking")
    graph.add_edge("generate_with_thinking", END)

    return graph.compile()


# ==============================================================================
# 7. BLOQUE DE EJECUCIÓN PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    # 1. Instanciamos el LegalRetriever REAL de la Fase 2 (src/retriever.py).
    #    Esto carga en memoria: chunks.jsonl, índice BM25, colección ChromaDB
    #    persistente, modelo de embeddings y Cross-Encoder. Requiere que la
    #    Fase 1 (chunker + indexer) se haya ejecutado previamente y que los
    #    artefactos existan en las rutas por defecto (data/chunks.jsonl,
    #    data/bm25_index.pkl, data/chroma_db).
    #    La inicialización se hace UNA sola vez, fuera del grafo, porque es
    #    costosa (carga de modelos); el grafo solo reutiliza esta instancia.
    retriever = LegalRetriever()

    # 2. Construimos y compilamos el grafo del agente.
    agent_graph = build_graph(retriever=retriever)

    # 3. Estado inicial: solo se rellena la query original.
    pregunta_usuario = "¿Quins són els deures dels voluntaris?"
    initial_state: AgentState = {
        "query": pregunta_usuario,
        "query_reescrita": "",
        "documentos": [],
        "respuesta_final": "",
    }

    print("=" * 80)
    print(f"PREGUNTA DE ENTRADA: {pregunta_usuario}")
    print("=" * 80)

    # 4. Ejecutamos el flujo completo del grafo.
    resultado_final: AgentState = agent_graph.invoke(initial_state)

    print("\n" + "=" * 80)
    print("QUERY REESCRITA:")
    print(resultado_final["query_reescrita"])

    print("\n" + "=" * 80)
    print("DOCUMENTOS RECUPERADOS:")
    for i, doc in enumerate(resultado_final["documentos"], start=1):
        print(f"  [{i}] {doc[:120]}...")

    print("\n" + "=" * 80)
    print("RESPUESTA FINAL:")
    print(resultado_final["respuesta_final"])
    print("=" * 80)