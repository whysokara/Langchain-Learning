from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

embedding = model.embed_query("who is Satoshi?")
print(embedding)