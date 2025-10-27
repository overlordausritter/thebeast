from fastapi import FastAPI, Request
from llama_cloud_services import LlamaCloudIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
import httpx
import asyncio
import os
import uvicorn

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

app = FastAPI()


def build_metadata_filters(filters_payload: dict | None) -> MetadataFilters | None:
    """
    Converts incoming Zapier 'filters' JSON into a LlamaIndex MetadataFilters object.
    Also adds %20-encoded variants for values with spaces to match URL encodings.
    """
    if not filters_payload or not isinstance(filters_payload, dict):
        return None

    filt_items = filters_payload.get("filters") or []
    if not filt_items:
        return None

    operator_map = {
        "==": FilterOperator.EQ,
        "eq": FilterOperator.EQ,
        "contains": FilterOperator.CONTAINS,
        "like": FilterOperator.LIKE,
        "in": FilterOperator.IN,
        "!=": FilterOperator.NE,
    }

    mf_list = []
    for f in filt_items:
        key = f.get("key")
        value = f.get("value")
        operator = f.get("operator", "==").lower()
        if not key or value is None:
            continue

        op_enum = operator_map.get(operator, FilterOperator.EQ)
        mf_list.append(MetadataFilter(key=key, operator=op_enum, value=value))

        # Add encoded-space variant for URLs (e.g., "Blue Ocean" → "Blue%20Ocean")
        if " " in value:
            encoded_value = value.replace(" ", "%20")
            mf_list.append(MetadataFilter(key=key, operator=op_enum, value=encoded_value))

    condition = filters_payload.get("condition")
    try:
        return MetadataFilters(filters=mf_list, condition=condition)
    except TypeError:
        return MetadataFilters(filters=mf_list)


@app.post("/llamaquery")
async def llamaquery(request: Request):
    data = await request.json()
    query = data.get("query")
    if not query:
        return {"error": "Missing 'query' in request body"}

    # Parse filters if provided by Zapier
    filters_payload = data.get("filters") or data.get("preFilters")
    metadata_filters = build_metadata_filters(filters_payload)

    # Create client and index
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    client = httpx.Client(timeout=timeout)

    index = LlamaCloudIndex(
        name="Sharepoint Deal Pipeline",
        project_name="The BEAST",
        organization_id="8ff953cd-9c16-49f2-93a4-732206133586",
        api_key=os.getenv("LLAMA_API_KEY"),
        client=client,
    )

    # Retry logic
    for attempt in range(3):
        try:
            retriever = index.as_retriever(filters=metadata_filters)
            nodes = await asyncio.to_thread(retriever.retrieve, query)
            break
        except (httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            if attempt < 2:
                print(f"Retry {attempt + 1}/3 after error: {e}")
                await asyncio.sleep(2)
                continue
            else:
                return {"error": f"Llama Cloud connection failed: {str(e)}"}

    # Combine text results
    text_output = "\n\n".join(
        [n.text for n in nodes if getattr(n, "text", None)]
    )

    # Collect unique citations
    citations = []
    seen = set()
    for node in nodes:
        node_obj = getattr(node, "node", node)
        metadata = getattr(node_obj, "metadata", {}) or {}

        file_name = metadata.get("file_name") or metadata.get("filename") or metadata.get("document_title")
        web_url = metadata.get("web_url")

        if (file_name or web_url) and (file_name, web_url) not in seen:
            citations.append({
                "file_name": file_name,
                "web_url": web_url,
            })
            seen.add((file_name, web_url))

    return {"text": text_output, "citations": citations}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
