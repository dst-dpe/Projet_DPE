# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from pathlib import Path

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
    st.title("🏠 Simulation DPE par Machine Learning")

    st.markdown(
        """
## Contexte
Ici tu présentes le sujet : DPE, enjeux, objectifs.

## Données
- Sources
- Variables (features)
- Target (ex: conso énergie / étiquette)

## Approche ML
- Préprocessing
- Modèles testés
- Métriques
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nb. lignes", "—")
    with col2:
        st.metric("Nb. variables", "—")
    with col3:
        st.metric("Score final", "—")

# ----------------------------
# PAGE 2: Dataviz
# ----------------------------
def page_dataviz():
    st.title("📊 Visualisation des Données")
    
    st.markdown("Cette section présente les résultats clés de l'analyse exploratoire réalisée en amont.")

    # --- Bloc 1 : Distribution ---
    st.header("1. Distribution des Étiquettes")
    st.markdown("Répartition des logements par classe énergétique (A à G).")
    
    # Assure-toi d'avoir une image nommée 'distrib_dpe.png' dans le dossier img/
    try:
        st.image("img/repartition_etiquette_DPE_France.png", caption="Répartition des classes DPE", use_container_width=True)
    except:
        st.warning("⚠️ Image 'img/repartition_etiquette_DPE_France.png' introuvable. Pense à l'ajouter dans ton repo !")

    # --- Bloc 2 : Carte ---
    st.header("2. Cartographie des Passoires Thermiques")
    st.markdown("Part des logements F et G par département.")
    
    try:
        st.image("img/part_passoires_thermiques_par_departement.png", caption="Géographie des passoires thermiques", use_container_width=True)
    except:
        st.warning("⚠️ Image 'img/part_passoires_thermiques_par_departement.png' introuvable.")

    # --- Bloc 3 : Autres Analyses ---
    st.header("3. Facteurs d'Influence")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Impact de la période de construction")
        st.markdown("Influence de l'année de construction sur la performance.")
        try:
            # Change le nom selon ce que tu as exporté
            st.image("img/repartition_etiquette_periode.png", use_container_width=True) 
        except:
            st.info("Ajoute 'repartition_etiquette_periode.png' pour voir ce graphique.")

    with c2:
        st.subheader("Impact de la surface")
        st.markdown("Répartition des surfaces par étiquette DPE")
        try:
            # Change le nom selon ce que tu as exporté
            st.image("img/surface_etiquette_boxplot.png", use_container_width=True)
        except:
            st.info("Ajoute 'surface_etiquette_boxplot.png' pour voir ce graphique.")

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
