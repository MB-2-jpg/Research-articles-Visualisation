import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 1. Load a text file containing multiple documents.
def load_documents(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    # Suppose that each document is separated by the delimiter "###"
    documents = [doc.strip() for doc in content.split("###") if doc.strip()]
    return documents

# 2. Apply TF-IDF transformation and LSA (using TruncatedSVD) to extract latent themes.
def perform_lsa(documents, n_components=2):
    # Using stop_words="english" (accepted by scikit-learn for English).
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(documents)
    
    # Apply truncated SVD (LSA)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_components = svd.fit_transform(X)
    
    # Extract feature names (terms)
    terms = vectorizer.get_feature_names_out()
    themes = {}
    for i, comp in enumerate(svd.components_):
        # Retrieve the 10 most important words for this component (theme)
        top_indices = comp.argsort()[:-11:-1]
        top_terms = [terms[j] for j in top_indices]
        themes[f"Theme {i+1}"] = top_terms
    
    return lsa_components, themes

# 3. Example usage:
filepath = "C:/Users/obeye/OneDrive/Bureau/Artishow/s2a_thèses_with_pdf.txt"  # Change this path to where your text file is located.
documents = load_documents(filepath)
lsa_components, themes = perform_lsa(documents, n_components=2)

print("Extracted Themes:")
for theme, top_terms in themes.items():
    print(f"{theme}: {top_terms}")

print("\nDocuments projection in the LSA latent space:")
print(lsa_components)


