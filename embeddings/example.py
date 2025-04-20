from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Textos base (como si fueran documentos)
documentos = [
    "Guía para entrenar una red neuronal desde cero.",
    "Recetas saludables con inteligencia artificial.",
    "Historia y evolución del machine learning.",
    "Cómo construir un modelo de IA paso a paso.",
    "Aplicaciones del deep learning en medicina."
]

# Crear vectorizador
vectorizador = TfidfVectorizer()

# Convertir documentos a embeddings
emb_docs = vectorizador.fit_transform(documentos)

# Texto de búsqueda (input del usuario)
consulta = input("🔍 Escribe una palabra o frase: ")

# Convertir consulta a embedding
emb_consulta = vectorizador.transform([consulta])

# Calcular similitud entre consulta y documentos
similitudes = cosine_similarity(emb_consulta, emb_docs)[0]

# Obtener el índice del documento más parecido
idx_mas_similar = similitudes.argmax()

#Imprimir la matriz
print(similitudes)

# Mostrar resultado
print("\n✅ Documento más parecido:")
print(f"→ {documentos[idx_mas_similar]}")
print(f"(Similitud: {similitudes[idx_mas_similar]:.2f})")

