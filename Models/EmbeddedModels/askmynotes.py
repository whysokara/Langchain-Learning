import warnings
warnings.filterwarnings("ignore")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# user_input = input("Add your notes here: ")
notes = [
    "In the beginning, data was stored in flat files.",
    "Apache Spark allows distributed data processing.",
    "The mitochondria is the powerhouse of the cell.",
    "Cricket is more than a sport in India; it's a passion."
]
embeddings = []
for note in notes:
    embedding = model.embed_query(note)
    embeddings.append(embedding)

user_question = input("Ask a question about your notes: ")
embedding_question = model.embed_query(user_question)

similarities = [cosine_similarity([e], [embedding_question])[0][0] for e in embeddings]

# Get index of highest similarity
most_similar_index = np.argmax(similarities)

# Print the most relevant note
print("Most relevant note:")
print(notes[most_similar_index])