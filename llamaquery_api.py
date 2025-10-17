from fastapi import FastAPI, Request
from llama_cloud_services import LlamaCloudIndex
import uvicorn

app = FastAPI()

@app.post("/llamaquery")
async def llamaquery(request: Request):
    data = await request.json()
    company_name = data.get("company_name", "")
    description = data.get("company_description", "")

    query = (
        f"Write a concise 3-paragraph business overview, description, "
        f"and brief history for {company_name}. "
        f"Use this context if helpful: {description}"
    )

    index = LlamaCloudIndex(
        name="Sharepoint Deal Pipeline",
        project_name="The BEAST",
        organization_id="8ff953cd-9c16-49f2-93a4-732206133586",
        api_key="llx-YOUR-KEY-HERE",
    )

    response = index.as_query_engine().query(query)
    return {"overview": str(response)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
