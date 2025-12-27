# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from pathlib import Path
import streamlit as st
import os

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="Simulateur DPE - Projet ML",
    page_icon="🏠",
    layout="wide",
)


# ----------------------------
# UTILS: chargements en cache
# ----------------------------
@st.cache_data(show_spinner=False)
def load_viz_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    # idéalement: un Pipeline sklearn qui inclut preprocessing + modèle
    return joblib.load(path)

# ----------------------------
# UI: Sidebar navigation
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller à :",
    [
        "🏁 Présentation",
        "📊 Dataviz",
        "📈 Résultats d'entraînement",
        "🧮 Simulateur DPE",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Projet ML - Simulation DPE")

# ----------------------------
# PAGE 1: Présentation
# ----------------------------
def page_presentation():
    # --- Sidebar : L'équipe ---
    with st.sidebar:
        st.markdown("### 👥 L'Équipe")
        st.markdown("""
        * **Yacine Bennouna**
        * **Aymane Karani**
        * **Dylan Nefnaf**
        * **Guillaume Deschamps**
        """)
        st.divider()
        st.info("Projet certifié Data Science")

    # --- En-tête Principal ---
    st.title("🏡 Projet DPE : Modélisation & Prédiction")
    
    st.markdown("""
    **Bienvenue sur l'interface de restitution de notre projet.**
    
    Ce projet explore les données du *Diagnostic de Performance Énergétique (DPE)* en France. 
    Il vise à appliquer des modèles de Machine Learning pour prédire l'étiquette énergétique des logements 
    et comprendre les facteurs déterminants de la consommation, à la croisée des enjeux techniques, économiques et scientifiques.
    """)

    st.divider()

    # --- Organisation en Onglets ---
    tab_contexte, tab_objectifs, tab_donnees = st.tabs(["🌍 Contexte & Réforme", "🎯 Objectifs", "💾 Données ADEME"])

    # --- ONGLET 1 : CONTEXTE ---
    with tab_contexte:
        st.header("Contexte Réglementaire et Technique")
        
        st.markdown("""
        Le DPE a subi une **réforme majeure le 1er juillet 2021** pour devenir un outil opposable et plus fiable. 
        Notre projet s'appuie exclusivement sur les données issues de ce nouveau cadre.
        """)

        with st.expander("⚖️ La Réforme DPE 2021 (Ce qui change)", expanded=True):
            st.markdown("""
            * **Méthode de calcul unifiée (3CL) :** Fin de la méthode "sur facture". Le calcul est désormais standardisé pour tous les logements.
            * **Double Seuil :** L'étiquette (A à G) est déterminée par la plus mauvaise note entre la **consommation d'énergie** et les **émissions de gaz à effet de serre (GES)**.
            * **5 Usages :** Prise en compte de l'éclairage et des auxiliaires (en plus du chauffage, de l'eau chaude et du refroidissement).
            """)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🛠️ Enjeu Technique")
            st.markdown("""
            La complexité réside dans la reproduction d'une méthode réglementaire stricte par des modèles statistiques.
            Le défi est de gérer des données hétérogènes (matériaux, systèmes) et de prédire une classe définie par des règles physiques.
            """)
        
        with col_b:
            st.markdown("#### 💰 Enjeu Économique")
            st.markdown("""
            Le DPE conditionne la valeur vénale ("Valeur Verte") et locative.
            L'objectif est d'aider à la décision pour prioriser les rénovations et anticiper les interdictions de location (passoires thermiques G+ dès 2023, G en 2025).
            """)

    # --- ONGLET 2 : OBJECTIFS ---
    with tab_objectifs:
        st.header("Objectifs du Projet")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("🤖 **Modélisation ML**")
            st.markdown("""
            * **Classification :** Prédire l'étiquette DPE (7 classes).
            * **Régression :** Estimer la consommation en kWh/m²/an.
            * **Comparaison :** Random Forest vs XGBoost vs Deep Learning.
            """)

        with col2:
            st.warning("📊 **Analyse & Biais**")
            st.markdown("""
            * **Facteurs clés :** Identifier les variables les plus influentes (Feature Importance).
            * **Déséquilibre :** Gérer la sous-représentation des classes extrêmes (A et G).
            * **Simplification :** Tester l'impact de la réduction des variables.
            """)

        with col3:
            st.success("🧠 **Interprétabilité**")
            st.markdown("""
            * **Explicabilité :** Utiliser SHAP pour comprendre les décisions du modèle.
            * **Critique :** Évaluer la capacité du ML à approximer une réglementation.
            * **Outil métier :** Proposer un simulateur interactif.
            """)

    # --- ONGLET 3 : DONNÉES (ENRICHI) ---
    with tab_donnees:
        st.header("Le Jeu de Données ADEME")
        
        # Métriques mises à jour avec les chiffres officiels
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Volume Total", "~13.6 Millions (12M au début du projet)", "DPE (Recensement continu)")
        col_m2.metric("Fréquence", "Hebdomadaire", "Mise à jour")
        col_m3.metric("Périmètre", "France", "Logements Existants")

        st.markdown("---")
        
        st.markdown("### 🔍 Spécificités du Dataset")
        st.warning("""
        **⚠️ Attention aux biais d'interprétation :**
        Selon l'ADEME, cette base n'est **pas représentative de l'ensemble du parc immobilier français**.
        Elle ne contient que les DPE réalisés obligatoirement lors de **ventes, locations ou constructions neuves**. 
        Un redressement statistique (croisement avec données INSEE) serait nécessaire pour une extrapolation nationale parfaite.
        """)

        st.markdown("""
        * **Source :** Base officielle [DPE Logements existants (depuis juillet 2021)](https://data.ademe.fr/datasets/dpe03existant).
        * **Contenu :** Caractéristiques techniques (bâti, isolation, chauffage), consommations énergétiques et émissions GES.
        * **Filtres appliqués pour le projet :**
            * Logements résidentiels uniquement (Maisons & Appartements).
            * Données nettoyées des valeurs aberrantes et doublons.
        """)
        
        st.caption("Données sous Licence Ouverte / Open Licence version 2.0 - Producteur : ADEME")
# ----------------------------
# PAGE 2: Dataviz
# ----------------------------


def display_img(filename, caption=""):
    """Fonction utilitaire pour gérer l'affichage sécurisé des images"""
    path = f"img/{filename}"
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"⚠️ Image manquante : {path}")

def page_dataviz():
    st.title("📊 Visualisation des Données DPE")
    st.markdown("""
    Cette section explore la répartition des classes énergétiques en France et analyse les corrélations 
    avec les caractéristiques physiques et géographiques des logements.
    """)

    # Création d'onglets pour organiser la navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 Panorama National", 
        "🗺️ Géographie & Climat", 
        "🏗️ Caractéristiques Bâti", 
        "⏳ Temps & Surface"
    ])

    # --- ONGLET 1 : PANORAMA NATIONAL ---
    with tab1:
        st.header("État des lieux du parc immobilier")
        
        st.markdown("### 1. Répartition DPE & GES")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Étiquette Énergie (DPE)**")
            display_img("repartition_etiquette_DPE_France.png", "Répartition nationale des DPE")
        with col2:
            st.markdown("**Étiquette Climat (GES)**")
            display_img("repartition_etiquette_GES_France.png", "Répartition nationale des GES")
            
        st.info("💡 **Note :** On observe souvent une corrélation entre les étiquettes DPE et GES, bien que le mode de chauffage influence fortement le GES.")

        st.markdown("### 2. Consommation réelle")
        display_img("repartition_conso_France.png", "Distribution de la consommation énergétique (kWh/m²/an)")

    # --- ONGLET 2 : GÉOGRAPHIE ---
    with tab2:
        st.header("Disparités Territoriales")
        
        st.markdown("### 1. La France des passoires vs bâtiments écolos")
        c1, c2 = st.columns(2)
        with c1:
            display_img("part_passoires_thermiques_par_departement.png", "Part des passoires (F & G)")
        with c2:
            display_img("part_batiments_ecolo_par_departements.png", "Part des bâtiments performants (A & B)")

        st.markdown("---")
        
        st.markdown("### 2. Influence de l'environnement")
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Par Région")
            display_img("repartition_DPE_regions.png", "DPE par Région administrative")
        with c4:
            st.subheader("Par Zone Climatique")
            display_img("repartition_zone_climatique.png", "Impact du climat local")
            
        st.markdown("#### Focus Altitude")
        display_img("repartition_classe_altitude.png", "Répartition des classes selon l'altitude")

    # --- ONGLET 3 : CARACTÉRISTIQUES BÂTI ---
    with tab3:
        st.header("Impact technique sur la performance")

        st.markdown("### 1. Type de bâtiment & Énergie")
        # Comparaison Maison vs Appartement (DPE & GES)
        c1, c2 = st.columns(2)
        with c1:
            display_img("etiquette_DPE_type_bat.png", "DPE selon le type de logement")
        with c2:
            display_img("etiquette_GES_type_bat.png", "GES selon le type de logement")
            
        st.markdown("#### Source d'énergie principale")
        display_img("repartition_type_energie_n1.png", "Répartition par type d'énergie")

        st.markdown("---")
        st.markdown("### 2. Inertie du bâtiment")
        st.markdown("L'inertie thermique joue un rôle clé dans le confort et la performance.")
        display_img("repartition_classe_inertie_batiment.png", "Classement selon l'inertie")

    # --- ONGLET 4 : TEMPS ET SURFACE ---
    with tab4:
        st.header("Construction et Dimensions")

        st.markdown("### 1. L'impact de l'ancienneté")
        st.markdown("L'évolution des normes de construction au fil du temps :")
        
        c1, c2 = st.columns(2)
        with c1:
            display_img("repartition_etiquette_periode.png", "Étiquettes par période de construction")
        with c2:
            display_img("repartition_periode_etiquette.png", "Périodes de construction par étiquette")

        st.markdown("---")

        st.markdown("### 2. L'impact de la surface")
        st.markdown("Les petites surfaces sont-elles défavorisées par le calcul du DPE ?")
        
        display_img("surface_etiquette_boxplot.png", "Distribution des surfaces par étiquette")

        with st.expander("🔎 Détail du nettoyage des données (Outliers)"):
            st.write("Analyse de la distribution des surfaces avant et après traitement des valeurs aberrantes.")
            col_a, col_b = st.columns(2)
            with col_a:
                display_img("surface_without_outliers.png", "Surface sans outliers")
            with col_b:
                display_img("surface_without_outliers_dist.png", "Distribution nettoyée")               
                
# ----------------------------
# PAGE 3: Résultats d'entraînement
# ----------------------------
def page_results():
    st.title("📈 Résultats d'entraînement")

    st.markdown(
        """
## Modèles testés
- Baseline
- RandomForest / XGBoost / NN
- Optimisation d'hyperparamètres

## Métriques
- MAE / RMSE / R² (si régression)
- Accuracy / F1 (si classification)

## Analyse d'erreur
- où le modèle se trompe le plus
- biais potentiels
        """
    )

    st.markdown("---")
    st.subheader("Illustrations / Courbes")
    st.info("Ici tu peux ajouter tes figures exportées (PNG) ou des courbes calculées à partir d'un CSV de logs.")

    # Exemple: afficher une image si tu en as
    # st.image("assets/loss_curve.png", caption="Courbe de loss", use_container_width=True)

# ----------------------------
# PAGE 4: Simulateur (Formulaire + Modèle)
# ----------------------------
def page_simulator():
    st.title("🧮 Simulateur DPE")
    st.write("Renseigne les caractéristiques du logement pour obtenir une estimation.")

    if not MODEL_PATH.exists():
        st.error(f"Modèle introuvable : {MODEL_PATH}")
        st.stop()

    model = load_model(MODEL_PATH)

    # ---- Définition des valeurs possibles (à adapter à ton dataset) ----
    # Idéalement: tu mets ces listes dans un fichier config (yaml/json) ou tu les derives du training.
    CATS = {
        "type_batiment": ["Maison", "Appartement"],
        "periode_construction": ["< 1948", "1949-1974", "1975-2000", "2001-2012", ">= 2013"],
        "qualite_isolation_murs": ["insuffisante", "moyenne", "bonne", "très bonne"],
        # ...
    }

    # ---- Formulaire ----
    with st.form("dpe_form"):
        st.subheader("Caractéristiques")

        c1, c2, c3 = st.columns(3)

        with c1:
            type_bat = st.selectbox("Type de bâtiment", CATS["type_batiment"])
            periode = st.selectbox("Période de construction", CATS["periode_construction"])

        with c2:
            surface = st.number_input("Surface (m²)", min_value=5.0, max_value=1000.0, value=60.0, step=1.0)
            hauteur = st.number_input("Hauteur sous plafond (m)", min_value=1.8, max_value=4.0, value=2.5, step=0.1)

        with c3:
            iso_murs = st.selectbox("Qualité isolation murs", CATS["qualite_isolation_murs"])
            # Ajoute d'autres champs...

        submitted = st.form_submit_button("Calculer le DPE")

    # ---- Inférence ----
    if submitted:
        # Construire une ligne au format modèle
        # IMPORTANT: les noms de colonnes doivent correspondre à ceux utilisés au training
        X = pd.DataFrame([{
            "type_batiment": type_bat,
            "periode_construction": periode,
            "surface_habitable": surface,
            "hauteur_sous_plafond": hauteur,
            "qualite_isolation_murs": iso_murs,
            # ...
        }])

        try:
            pred = model.predict(X)

            # Si ton modèle renvoie un scalaire
            y = float(np.ravel(pred)[0])

            st.success("Résultat calculé ✅")
            st.metric("Estimation (valeur)", f"{y:,.2f}")

            # Option: transformer en étiquette DPE si tu as un mapping
            # etiquette = to_dpe_label(y)
            # st.metric("Étiquette DPE", etiquette)

            with st.expander("Voir les données envoyées au modèle"):
                st.dataframe(X, use_container_width=True)

        except Exception as e:
            st.error("Erreur lors du calcul. Vérifie la compatibilité features / preprocessing.")
            st.exception(e)

# ----------------------------
# ROUTER
# ----------------------------
if page == "🏁 Présentation":
    page_presentation()
elif page == "📊 Dataviz":
    page_dataviz()
elif page == "📈 Résultats d'entraînement":
    page_results()
elif page == "🧮 Simulateur DPE":
    page_simulator()
