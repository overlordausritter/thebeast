from fastapi import FastAPI, Request
from llama_cloud_services import LlamaCloudIndex
import uvicorn
import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

app = FastAPI()

@app.post("/llamaquery")
async def llamaquery(request: Request):
    data = await request.json()
    query = data.get("query")

    index = LlamaCloudIndex(
        name="Sharepoint Deal Pipeline",
        project_name="The BEAST",
        organization_id="8ff953cd-9c16-49f2-93a4-732206133586",
        api_key=os.getenv("LLAMA_API_KEY"),
    )

    nodes = index.as_retriever().retrieve(query)

    # just return the raw text from all chunks, concatenated
    text_output = "\n\n".join([n.text for n in nodes if n.text])

    return {"text": text_output}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
