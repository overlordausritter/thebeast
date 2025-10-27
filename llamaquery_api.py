from fastapi import FastAPI, Request
from llama_cloud_services import LlamaCloudIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
import httpx
import asyncio
import os
import uvicorn

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

app = FastAPI()


def normalize_variants(value: str) -> list[str]:
    """
    Generate normalized variants for space encodings and case differences.
    Example: 'Blue Ocean' → ['blue ocean', 'blue%20ocean', 'blue_ocean', 'blueocean']
    """
    if not value:
        return []
    v = value.strip()
    lower = v.lower()
    variants = {lower, lower.replace(" ", "%20"), lower.replace(" ", "_"), lower.replace(" ", "")}
    return list(variants)


@app.post("/llamaquery")
async def llamaquery(request: Request):
    data = await request.json()
    query = data.get("query")
    if not query:
        return {"error": "Missing 'query' in request body"}

    # Extract the company name from filters sent by Zapier
    filters_payload = data.get("filters") or data.get("preFilters")
    company_value = None
    if filters_payload and "filters" in filters_payload:
        first = filters_payload["filters"][0]
        company_value = first.get("value")

    if not company_value:
        return {"error": "Missing 'company name' filter value"}

    # Generate normalized variants for the company name
    variants = normalize_variants(company_value)

    # Build metadata filters with OR condition
    mf_list = []
    for val in variants:
        mf_list.extend([
            MetadataFilter(key="file_name", operator=FilterOperator.CONTAINS, value=val),
            MetadataFilter(key="web_url", operator=FilterOperator.CONTAINS, value=val),
        ])
    filters = MetadataFilters(filters=mf_list, condition="or")

    # Configure client and LlamaCloud index
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
            retriever = index.as_retriever(filters=filters)
            nodes = await asyncio.to_thread(retriever.retrieve, query)
            break
        except (httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            if attempt < 2:
                print(f"Retry {attempt + 1}/3 after error: {e}")
                await asyncio.sleep(2)
                continue
            else:
                return {"error": f"Llama Cloud connection failed: {str(e)}"}

    # No post-filter enforcement — rely on retriever filtering only
    if not nodes:
        return {"text": f"No matching documents found for '{company_value}'.", "citations": []}

    text_output = "\n\n".join([n.text for n in nodes if getattr(n, "text", None)])

    citations = []
    seen = set()
    for node in nodes:
        node_obj = getattr(node, "node", node)
        metadata = getattr(node_obj, "metadata", {}) or {}
        file_name = metadata.get("file_name") or metadata.get("filename") or metadata.get("document_title")
        web_url = metadata.get("web_url")
        if (file_name or web_url) and (file_name, web_url) not in seen:
            citations.append({"file_name": file_name, "web_url": web_url})
            seen.add((file_name, web_url))

    return {"text": text_output, "citations": citations}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
