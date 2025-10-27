from fastapi import FastAPI, Request
from llama_cloud_services import LlamaCloudIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
import httpx
import asyncio
import os
import uvicorn

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

app = FastAPI()


@app.post("/llamaquery")
async def llamaquery(request: Request):
    """Company-scoped retrieval endpoint using LlamaCloud metadata filters."""
    data = await request.json()
    query = data.get("query")
    if not query:
        return {"error": "Missing 'query' in request body"}

    # Extract the company name from Zapier payload
    company_name = None
    try:
        filters_payload = data.get("filters") or data.get("preFilters")
        if filters_payload and "filters" in filters_payload:
            first_filter = filters_payload["filters"][0]
            company_name = first_filter.get("value")
    except Exception:
        company_name = None

    if not company_name:
        return {"error": "Missing company name in request payload"}

    # Build LlamaCloud metadata filter using EQ operator (recommended pattern)
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="company",  # <-- must match the metadata key used during ingestion
                operator=FilterOperator.EQ,
                value=company_name,
            ),
        ]
    )

    # Custom HTTP client with extended timeouts
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    client = httpx.Client(timeout=timeout)

    # Initialize the LlamaCloud index
    index = LlamaCloudIndex(
        name="Sharepoint Deal Pipeline",
        project_name="The BEAST",
        organization_id="8ff953cd-9c16-49f2-93a4-732206133586",
        api_key=os.getenv("LLAMA_API_KEY"),
        client=client,
    )

    # Retry logic for transient network issues
    for attempt in range(3):
        try:
            retriever = index.as_retriever(
                dense_similarity_top_k=3,
                sparse_similarity_top_k=3,
                alpha=0.5,
                enable_reranking=True,
                rerank_top_n=3,
                filters=filters,
            )
            nodes = await asyncio.to_thread(retriever.retrieve, query)
            break
        except (httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            if attempt < 2:
                print(f"Retry {attempt + 1}/3 after error: {e}")
                await asyncio.sleep(2)
                continue
            else:
                return {"error": f"Llama Cloud connection failed: {str(e)}"}

    # If no nodes were returned, surface a clear message
    if not nodes:
        return {
            "text": f"No documents found for company '{company_name}'.",
            "citations": [],
        }

    # Combine retrieved text
    text_output = "\n\n".join(
        [n.text for n in nodes if getattr(n, "text", None)]
    )

    # Collect unique citations
    citations = []
    seen = set()
    for node in nodes:
        node_obj = getattr(node, "node", node)
        metadata = getattr(node_obj, "metadata", {}) or {}
        file_name = (
            metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("document_title")
        )
        web_url = metadata.get("web_url")
        if (file_name or web_url) and (file_name, web_url) not in seen:
            citations.append({"file_name": file_name, "web_url": web_url})
            seen.add((file_name, web_url))

    return {"text": text_output, "citations": citations}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
