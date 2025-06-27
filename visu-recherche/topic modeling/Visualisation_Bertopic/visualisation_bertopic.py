# -*- coding: utf-8 -*-

#à corriger
"""
BERTopic avec Dash pour une vraie interactivité style pyLDAvis
Installation requise: pip install bertopic dash
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import pandas as pd
from sklearn.manifold import MDS
import plotly.graph_objects as go
import plotly.express as px
from bertopic import BERTopic
import dash
from dash import dcc, html, Input, Output, callback
import webbrowser
from threading import Timer

def conversion_txt_string(nom_fichier):
    try:
        with open(nom_fichier, "r", encoding="utf-8") as fichier:
            contenu = fichier.read()
        return contenu
    except FileNotFoundError:
        print(f"Erreur : fichier '{nom_fichier}' non trouvé.")
        return None
    except Exception as e:
        print(f"Erreur : {e}")
        return None

def prepare_bertopic_model():
    """Prépare le modèle BERTopic avec des paramètres optimisés"""
    
    # Modèle d'embedding
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # Réduction dimensionnelle UMAP
    umap_model = umap.UMAP(
        n_neighbors=15, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine',
        random_state=42
    )
    
    # Clustering HDBSCAN
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=2,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Vectorizer pour les mots-clés
    custom_stopwords = list(text.ENGLISH_STOP_WORDS.union([
        'le', 'la', 'les', 'de', 'des', 'du', 'un', 'une', 'et', 'en', 'dans', 'au', 'aux', 'ce', 'ces', 'ça',
        'pour', 'pas', 'par', 'sur', 'se', 'plus', 'ou', 'avec', 'tout', 'mais', 'comme', 'si', 'sans', 'être',
        'cette', 'son', 'sa', 'ses', 'on', 'il', 'elle', 'ils', 'elles', 'nous', 'vous', 'je', 'tu', 'mon', 'ma',
        'mes', 'ton', 'ta', 'tes', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs', 'y', 'donc', 'hal', 'thèse',
        'université', 'recherche', 'étude', 'pdf', 'contenu', 'ainsi', 'après', 'avant', 'bien', 'cela'
    ]))
    
    vectorizer_model = TfidfVectorizer(
        stop_words=custom_stopwords,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )
    
    # Modèle BERTopic complet
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="multilingual",
        calculate_probabilities=True,
        verbose=True
    )
    
    return topic_model

def calculate_topic_positions(topic_model, embeddings, topics):
    """Calcule les positions 2D des topics"""
    
    topic_embeddings = {}
    
    for topic_id in set(topics):
        if topic_id == -1:  # Ignorer outliers
            continue
        
        topic_indices = [i for i, t in enumerate(topics) if t == topic_id]
        
        if len(topic_indices) > 0:
            topic_docs_embeddings = embeddings[topic_indices]
            centroid = np.mean(topic_docs_embeddings, axis=0)
            topic_embeddings[topic_id] = centroid
    
    # MDS pour réduction 2D
    topic_ids = list(topic_embeddings.keys())
    if len(topic_ids) <= 1:
        return {topic_ids[0]: {'x': 0, 'y': 0}} if topic_ids else {}
    
    centroids_matrix = np.array([topic_embeddings[tid] for tid in topic_ids])
    
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
    coords_2d = mds.fit_transform(centroids_matrix)
    
    topic_coords = {}
    for i, topic_id in enumerate(topic_ids):
        topic_coords[topic_id] = {
            'x': coords_2d[i, 0],
            'y': coords_2d[i, 1]
        }
    
    return topic_coords

# Variables globales pour Dash
topic_model = None
topic_data_global = {}
documents_global = []

def prepare_data():
    """Prépare toutes les données pour l'application Dash"""
    global topic_model, topic_data_global, documents_global
    
    print("=== PRÉPARATION DES DONNÉES BERTOPIC ===")
    
    # 1. Chargement des données
    print("[1/5] Chargement des données...")
    content = conversion_txt_string("s2a_thèses_with_pdf.txt")
    if content is None:
        return False
    
    documents = content.split("\n\n\n\n\n Thèse ")[1:]
    documents = [doc.strip() for doc in documents if len(doc.strip()) > 100]
    documents_global = documents
    
    
    print(f"Nombre de documents: {len(documents)}")
    
    if len(documents) < 2:
        print("Erreur: Pas assez de documents")
        return False
    
    # 2. Modèle BERTopic
    print("[2/5] Création du modèle BERTopic...")
    topic_model = prepare_bertopic_model()
    
    # 3. Garder une référence au modèle d'embedding original
    embedding_model_original = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # 4. Entraînement
    print("[3/5] Entraînement BERTopic...")
    topics, probabilities = topic_model.fit_transform(documents)
    
    # DEBUG: Analyser les topics trouvés
    print(f"\n=== DEBUG TOPICS ===")
    unique_topics = set(topics)
    print(f"Topics uniques trouvés: {sorted(unique_topics)}")
    for topic in sorted(unique_topics):
        count = topics.count(topic)
        print(f"Topic {topic}: {count} documents")
    
    # Si trop de documents sont outliers, ajuster les paramètres
    outliers_count = topics.count(-1)
    valid_topics_count = len(unique_topics) - (1 if -1 in unique_topics else 0)
    
    print(f"Documents outliers (topic -1): {outliers_count}")
    print(f"Topics valides: {valid_topics_count}")
    
    if valid_topics_count < 3 or outliers_count > len(documents) * 0.7:
        print("⚠️  Trop peu de topics ou trop d'outliers, réajustement des paramètres...")
        
        # Créer un modèle plus permissif
        hdbscan_permissive = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,  # Plus permissif
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        topic_model_permissive = BERTopic(
            embedding_model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
            umap_model=umap.UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine'),
            hdbscan_model=hdbscan_permissive,
            vectorizer_model=vectorizer_model,
            language="multilingual",
            verbose=True
        )
        
        print("Ré-entraînement avec paramètres ajustés...")
        topics, probabilities = topic_model_permissive.fit_transform(documents)
        topic_model = topic_model_permissive
        
        # DEBUG après ajustement
        unique_topics = set(topics)
        print(f"Topics après ajustement: {sorted(unique_topics)}")
        for topic in sorted(unique_topics):
            count = topics.count(topic)
            print(f"Topic {topic}: {count} documents")
    
    # 5. Calcul des embeddings avec le modèle original
    print("[4/5] Calcul des embeddings...")
    embeddings = embedding_model_original.encode(documents)
    
    # 6. Calcul des positions des topics
    print("[5/5] Calcul des positions des topics...")
    topic_coords = calculate_topic_positions(topic_model, embeddings, topics)
    
    # 7. Préparation des données globales
    topic_info = topic_model.get_topic_info()
    print(f"\n=== DEBUG TOPIC_INFO ===")
    print(f"Nombre de lignes dans topic_info: {len(topic_info)}")
    print("Aperçu topic_info:")
    print(topic_info[['Topic', 'Count', 'Name']])
    
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            print(f"Ignoré topic {topic_id} (outliers)")
            continue
        
        topic_words = topic_model.get_topic(topic_id)
        coords = topic_coords.get(topic_id, {'x': 0, 'y': 0})
        
        print(f"Topic {topic_id}: {len(topic_words)} mots, position ({coords['x']:.2f}, {coords['y']:.2f})")
        
        topic_data_global[topic_id] = {
            'x': coords['x'],
            'y': coords['y'],
            'size': row['Count'],
            'words': [word for word, score in topic_words[:10]],
            'scores': [score for word, score in topic_words[:10]],
            'keywords_str': ', '.join([word for word, score in topic_words[:5]])
        }
    
    print(f"Topics ajoutés à topic_data_global: {list(topic_data_global.keys())}")
    print(f"Nombre total de topics pour visualisation: {len(topic_data_global)}")
    
    return True

def create_circles_plot():
    """Crée le graphique des cercles (topics)"""
    
    print(f"\n=== DEBUG CRÉATION CERCLES ===")
    print(f"topic_data_global contient: {len(topic_data_global)} topics")
    print(f"Topics: {list(topic_data_global.keys())}")
    
    if not topic_data_global:
        print("⚠️  Aucune donnée de topic disponible!")
        # Créer un graphique vide avec un message
        fig = go.Figure()
        fig.add_annotation(
            text="Aucun topic trouvé<br>Vérifiez les paramètres BERTopic",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Aucun topic détecté",
            xaxis_title="Composante 1",
            yaxis_title="Composante 2",
            height=500
        )
        return fig
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    max_size = max([data['size'] for data in topic_data_global.values()])
    print(f"Taille max de topic: {max_size}")
    
    for i, (topic_id, data) in enumerate(topic_data_global.items()):
        # Normaliser la taille (minimum 20, maximum 100)
        normalized_size = max(20, (data['size'] / max_size) * 80 + 30)
        
        print(f"Topic {topic_id}: position=({data['x']:.2f}, {data['y']:.2f}), taille={data['size']}, taille_normalisée={normalized_size:.1f}")
        
        fig.add_trace(go.Scatter(
            x=[data['x']],
            y=[data['y']],
            mode='markers+text',
            marker=dict(
                size=normalized_size,
                color=colors[i % len(colors)],
                opacity=0.7,
                line=dict(width=2, color='black')
            ),
            text=[str(topic_id)],
            textfont=dict(size=14, color='black'),
            name=f'Topic {topic_id}',
            hovertemplate=
            f'<b>Topic {topic_id}</b><br>' +
            f'Taille: {data["size"]} documents<br>' +
            f'Mots-clés: {data["keywords_str"]}<br>' +
            '<extra></extra>',
            customdata=[topic_id]  # Pour les callbacks
        ))
    
    fig.update_layout(
        title=f"Distribution des Topics BERTopic ({len(topic_data_global)} topics trouvés)<br>Cliquez sur un cercle",
        xaxis_title="Composante 1",
        yaxis_title="Composante 2",
        showlegend=False,
        height=500,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    
    print("Graphique des cercles créé avec succès!")
    return fig

def create_keywords_plot(topic_id):
    """Crée le graphique des mots-clés pour un topic donné"""
    
    if topic_id not in topic_data_global:
        return go.Figure()
    
    data = topic_data_global[topic_id]
    words = data['words'][:10]
    scores = data['scores'][:10]
    
    # Inverser pour avoir le plus important en haut
    words_rev = words[::-1]
    scores_rev = scores[::-1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=words_rev,
        x=scores_rev,
        orientation='h',
        marker_color='lightblue',
        text=[f'{score:.3f}' for score in scores_rev],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Mots-clés du Topic {topic_id}",
        xaxis_title="Score c-TF-IDF",
        yaxis_title="Mots-clés",
        height=500,
        margin=dict(l=100)
    )
    
    return fig

# APPLICATION DASH
app = dash.Dash(__name__)

# Layout initial vide - sera mis à jour après chargement des données
app.layout = html.Div([
    html.H1("BERTopic - Visualisation Interactive Style pyLDAvis", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div(id='main-content', children=[
        html.Div("Chargement des données en cours...", 
                style={'textAlign': 'center', 'fontSize': 20, 'margin': 50})
    ]),
    
    # Store pour déclencher le chargement
    dcc.Store(id='data-loaded', data=False),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0, max_intervals=1)
])

@app.callback(
    Output('data-loaded', 'data'),
    Input('interval-component', 'n_intervals')
)
def load_data_callback(n):
    """Callback pour charger les données au démarrage"""
    if n > 0:
        return prepare_data()
    return False

@app.callback(
    Output('main-content', 'children'),
    Input('data-loaded', 'data')
)
def update_layout(data_loaded):
    """Met à jour le layout après chargement des données"""
    
    if not data_loaded:
        return html.Div("Erreur lors du chargement des données", 
                       style={'textAlign': 'center', 'color': 'red'})
    
    if not topic_data_global:
        return html.Div("Aucun topic trouvé dans les données", 
                       style={'textAlign': 'center', 'color': 'orange'})
    
    # Créer le layout principal avec les données chargées
    return html.Div([
        html.Div([
            # Graphique des cercles (gauche)
            html.Div([
                dcc.Graph(
                    id='circles-plot',
                    figure=create_circles_plot(),
                    style={'height': '500px'}
                )
            ], style={'width': '60%', 'display': 'inline-block'}),
            
            # Graphique des mots-clés (droite)
            html.Div([
                dcc.Graph(
                    id='keywords-plot',
                    figure=create_keywords_plot(list(topic_data_global.keys())[0]),
                    style={'height': '500px'}
                )
            ], style={'width': '40%', 'display': 'inline-block'})
        ]),
        
        # Informations sur le topic sélectionné
        html.Div(id='topic-info', children=create_default_topic_info(), style={
            'marginTop': 20, 
            'padding': 20, 
            'backgroundColor': '#f0f0f0', 
            'borderRadius': 5
        }),
        
        # Tableau récapitulatif
        html.Div([
            html.H3("Résumé des Topics"),
            html.Div(id='topics-summary', children=create_topics_summary())
        ], style={'marginTop': 30})
    ])

def create_default_topic_info():
    """Crée les informations par défaut du premier topic"""
    if topic_data_global:
        first_topic = list(topic_data_global.keys())[0]
        data = topic_data_global[first_topic]
        return html.Div([
            html.H4(f"Topic {first_topic} sélectionné"),
            html.P(f"Nombre de documents: {data['size']}"),
            html.P(f"Mots-clés principaux: {data['keywords_str']}"),
            html.P(f"Position: ({data['x']:.2f}, {data['y']:.2f})")
        ])
    return html.P("Sélectionnez un topic")

def create_topics_summary():
    """Crée le résumé des topics"""
    if not topic_data_global:
        return "Aucune donnée disponible"
    
    total_docs = sum([data['size'] for data in topic_data_global.values()])
    
    summary_items = []
    for topic_id, data in sorted(topic_data_global.items()):
        percentage = (data['size'] / total_docs) * 100
        summary_items.append(
            html.P(f"Topic {topic_id}: {data['keywords_str']} "
                  f"({data['size']} docs, {percentage:.1f}%)")
        )
    
    return summary_items

@app.callback(
    [Output('keywords-plot', 'figure'),
     Output('topic-info', 'children')],
    [Input('circles-plot', 'clickData')],
    prevent_initial_call=True
)
def update_keywords_plot(clickData):
    """Met à jour le graphique des mots-clés quand on clique sur un cercle"""
    
    if clickData is None:
        # Afficher le premier topic par défaut
        topic_id = list(topic_data_global.keys())[0] if topic_data_global else 0
    else:
        # Récupérer le topic cliqué
        topic_id = clickData['points'][0]['customdata']
    
    # Mettre à jour le graphique des mots-clés
    keywords_fig = create_keywords_plot(topic_id)
    
    # Informations sur le topic
    if topic_id in topic_data_global:
        data = topic_data_global[topic_id]
        info = html.Div([
            html.H4(f"Topic {topic_id} sélectionné"),
            html.P(f"Nombre de documents: {data['size']}"),
            html.P(f"Mots-clés principaux: {data['keywords_str']}"),
            html.P(f"Position: ({data['x']:.2f}, {data['y']:.2f})")
        ])
    else:
        info = html.P("Sélectionnez un topic en cliquant sur un cercle")
    
    return keywords_fig, info

def open_browser():
    """Ouvre le navigateur automatiquement"""
    webbrowser.open_new("http://localhost:8050/")

if __name__ == '__main__':
    print("=== LANCEMENT BERTOPIC DASH ===")
    print("🚀 Démarrage de l'application...")
    print("📱 L'application s'ouvrira automatiquement dans votre navigateur")
    print("🔗 URL: http://localhost:8050/")
    print("\n" + "="*50)
    print("INSTRUCTIONS:")
    print("1. Attendez le chargement des données (BERTopic)")
    print("2. Cliquez sur les cercles pour voir les mots-clés")
    print("3. La taille des cercles = nombre de documents")
    print("4. Position = similarité entre topics")
    print("5. Fermez avec Ctrl+C")
    print("="*50)
    
    # Ouvrir le navigateur après 2 secondes
    Timer(2.0, open_browser).start()
    
    # Lancer l'application
    app.run(debug=False, use_reloader=False)