from fastapi import FastAPI, Request
from llama_cloud_services import LlamaCloudIndex
import httpx
import asyncio
import os
import uvicorn

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

app = FastAPI()


@app.post("/llamaquery")
async def llamaquery(request: Request):
    data = await request.json()
    query = data.get("query")
    if not query:
        return {"error": "Missing 'query' in request body"}

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
            nodes = await asyncio.to_thread(index.as_retriever().retrieve, query)
            break
        except (httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            if attempt < 2:
                print(f"Retry {attempt + 1}/3 after error: {e}")
                await asyncio.sleep(2)
                continue
            else:
                return {"error": f"Llama Cloud connection failed: {str(e)}"}

    # Build plain text output exactly as before
    text_output = "\n\n".join(
        [n.text for n in nodes if getattr(n, "text", None)]
    )

    # Extract file_name and web_url for citations
    citations = []
    seen = set()
    for node in nodes:
        node_obj = getattr(node, "node", node)
        metadata = getattr(node_obj, "metadata", {}) or {}

        file_name = metadata.get("file_name") or metadata.get("filename") or metadata.get("document_title")
        web_url = metadata.get("web_url")  # Use web_url explicitly

        if (file_name or web_url) and (file_name, web_url) not in seen:
            citations.append({
                "file_name": file_name,
                "web_url": web_url,
            })
            seen.add((file_name, web_url))

    return {
        "text": text_output,
        "citations": citations
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
