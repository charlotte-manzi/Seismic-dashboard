"""
Analyse des Caractéristiques Sismiques - VERSION CORRIGÉE

Ce module fournit une analyse complète des caractéristiques sismiques incluant :
- Distribution des magnitudes avec catégorisation
- Distribution des profondeurs et impact sur les dégâts
- Relations entre magnitude et profondeur
- Calcul et analyse du potentiel destructeur
- Analyse de l'énergie libérée
- Tests statistiques avancés

Converti depuis Jupyter notebook vers Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
import sys
import os

# Supprimer les avertissements
warnings.filterwarnings('ignore')

# Ajouter utils au chemin pour le chargement des données
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Configuration matplotlib
plt.style.use('default')
sns.set_palette("husl")

def apply_custom_css():
    """Appliquer le CSS personnalisé pour l'analyse des caractéristiques"""
    st.markdown("""
    <style>
    .characteristics-header {
        background: linear-gradient(135deg, #e74c3c 0%, #8e44ad 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .intro-section {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .intro-text {
        color: #155724;
        font-size: 16px;
        line-height: 1.6;
        margin: 0;
        text-align: center;
    }
    
    .analysis-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #e74c3c;
    }
    
    .stats-container {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2ecc71;
        color: #155724;
        font-weight: 500;
    }
    
    .stats-container h4 {
        color: #155724;
        margin-bottom: 10px;
        font-weight: bold;
    }
    
    .stats-container p {
        color: #155724;
        margin: 5px 0;
        font-weight: 500;
    }
    
    .energy-metric {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
        color: #856404;
        font-weight: 500;
    }
    
    .energy-metric h4 {
        color: #856404;
        margin-bottom: 10px;
        font-weight: bold;
    }
    
    .energy-metric p {
        color: #856404;
        margin: 5px 0;
        font-weight: 500;
    }
    
    .danger-alert {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #dc3545;
        color: #721c24;
        font-weight: 500;
    }
    
    .danger-alert h4 {
        color: #721c24;
        margin-bottom: 10px;
        font-weight: bold;
    }
    
    .danger-alert p {
        color: #721c24;
        margin: 5px 0;
        font-weight: 500;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def show_analyse_caracteristiques():
    """Fonction principale pour afficher l'analyse des caractéristiques"""
    
    # Appliquer le style personnalisé
    apply_custom_css()
    
    # En-tête
    st.markdown("""
    <div class="characteristics-header">
        <h1>🔬 Analyse des Caractéristiques Sismiques</h1>
        <p>Analyse approfondie des propriétés physiques des séismes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section d'introduction
    st.markdown("""
    <div class="intro-section">
        <p style="text-align: center; font-weight: bold; line-height: 1.8; color: #155724; font-size: 16px; margin: 0;">
            ✅ <strong>15407 séismes réels chargés</strong><br><br>
            Ce module permet d'analyser les <strong>caractéristiques physiques</strong> des séismes. 
            Explorez la <strong>distribution des magnitudes</strong>, <strong>profondeurs</strong>, 
            le <strong>potentiel destructeur</strong> et l'<strong>énergie libérée</strong>. 
            📊 Sélectionnez un type d'analyse ci-dessous pour commencer votre exploration.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Obtenir les données filtrées depuis l'état de session
    if 'filtered_df' not in st.session_state:
        st.error("❌ Données non disponibles. Veuillez retourner à la page d'accueil.")
        return
    
    df = st.session_state.filtered_df
    
    if len(df) == 0:
        st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés.")
        return
    
    # Préparer les données avec les caractéristiques calculées
    df = prepare_seismic_characteristics(df)
    
    # Section de sélection du type d'analyse
    st.subheader("🔍 Sélection du Type d'Analyse")
    
    analysis_type = st.selectbox(
        "Choisissez le type d'analyse des caractéristiques :",
        [
            "Distribution des magnitudes",
            "Distribution des profondeurs", 
            "Relation magnitude/profondeur",
            "Potentiel destructeur",
            "Énergie libérée"
        ],
        index=0,
        help="Sélectionnez le type d'analyse physique à effectuer"
    )
    
    # Filtres avancés spécifiques aux caractéristiques
    show_advanced_filters(df)
    
    # Afficher les métriques clés
    show_key_metrics(df)
    
    # Exécuter l'analyse sélectionnée
    if analysis_type == "Distribution des magnitudes":
        analyser_distribution_magnitudes(df)
    elif analysis_type == "Distribution des profondeurs":
        analyser_distribution_profondeurs(df)
    elif analysis_type == "Relation magnitude/profondeur":
        analyser_relation_magnitude_profondeur(df)
    elif analysis_type == "Potentiel destructeur":
        analyser_potentiel_destructeur(df)
    elif analysis_type == "Énergie libérée":
        analyser_energie(df)

def prepare_seismic_characteristics(df):
    """Préparer les caractéristiques sismiques calculées - VERSION CORRIGÉE"""
    
    df = df.copy()
    
    # Correction des profondeurs négatives
    if (df['Profondeur'] < 0).any():
        st.warning(f"⚠️ {(df['Profondeur'] < 0).sum()} valeurs de profondeur négatives détectées. Application de la valeur absolue.")
        df['Profondeur'] = df['Profondeur'].abs()
    
    # CORRECTION : Traiter les profondeurs nulles ou très faibles
    # Remplacer les profondeurs nulles ou négatives par 1 km (valeur minimale raisonnable)
    df['Profondeur'] = df['Profondeur'].replace(0, 1.0)  # Remplacer 0 par 1 km
    df['Profondeur'] = np.where(df['Profondeur'] < 0.1, 1.0, df['Profondeur'])  # Minimum 0.1 km
    
    # Vérifier et corriger les valeurs NaN dans Magnitude et Profondeur
    if df['Magnitude'].isna().any():
        st.warning(f"⚠️ {df['Magnitude'].isna().sum()} valeurs de magnitude manquantes détectées. Suppression des lignes.")
        df = df.dropna(subset=['Magnitude'])
    
    if df['Profondeur'].isna().any():
        st.warning(f"⚠️ {df['Profondeur'].isna().sum()} valeurs de profondeur manquantes détectées. Suppression des lignes.")
        df = df.dropna(subset=['Profondeur'])
    
    # Catégorisation des magnitudes
    def categorize_magnitude(mag):
        if pd.isna(mag):
            return 'Inconnu'
        if 0 <= mag < 2.5:
            return 'Micro'
        elif 2.5 <= mag < 4.0:
            return 'Faible'
        elif 4.0 <= mag < 5.0:
            return 'Léger'
        elif 5.0 <= mag < 6.0:
            return 'Modéré'
        elif 6.0 <= mag < 7.0:
            return 'Fort'
        elif 7.0 <= mag < 8.0:
            return 'Majeur'
        elif mag >= 8.0:
            return 'Grand'
        return 'Inconnu'
    
    df['Magnitude_Categorie'] = df['Magnitude'].apply(categorize_magnitude)
    
    # Catégorisation des profondeurs
    def categorize_depth(depth):
        if pd.isna(depth):
            return 'Inconnu'
        if 0 <= depth < 70:
            return 'Peu profond'
        elif 70 <= depth < 300:
            return 'Intermédiaire'
        elif depth >= 300:
            return 'Profond'
        return 'Inconnu'
    
    df['Profondeur_Categorie'] = df['Profondeur'].apply(categorize_depth)
    
    # Calcul de l'énergie libérée (formule: E = 10^(1.5*M+4.8))
    # CORRECTION : Vérifier les valeurs avant le calcul
    df['Energie'] = np.where(
        df['Magnitude'].notna() & (df['Magnitude'] >= 0),
        10**(1.5 * df['Magnitude'] + 4.8),
        np.nan
    )
    
    # Calcul du potentiel destructeur - VERSION CORRIGÉE
    # Formule: Magnitude * (1 + 70/profondeur)
    # CORRECTION : S'assurer qu'il n'y a pas de division par zéro et limiter les valeurs extrêmes
    df['Potentiel_Destructeur'] = np.where(
        (df['Magnitude'].notna()) & (df['Profondeur'].notna()) & (df['Profondeur'] > 0),
        df['Magnitude'] * (1 + np.minimum(70/df['Profondeur'], 1000)),  # Limiter à 1000 max
        np.nan
    )
    
    # Supprimer les valeurs infinies ou NaN du potentiel destructeur
    if df['Potentiel_Destructeur'].isna().any():
        nb_nan = df['Potentiel_Destructeur'].isna().sum()
        st.warning(f"⚠️ {nb_nan} valeurs de potentiel destructeur invalides supprimées.")
        df = df.dropna(subset=['Potentiel_Destructeur'])
    
    # Vérifier les valeurs infinies
    if np.isinf(df['Potentiel_Destructeur']).any():
        nb_inf = np.isinf(df['Potentiel_Destructeur']).sum()
        st.warning(f"⚠️ {nb_inf} valeurs infinies de potentiel destructeur détectées et supprimées.")
        df = df[~np.isinf(df['Potentiel_Destructeur'])]
    
    # Catégorisation du potentiel destructeur
    def categorize_potentiel(pot):
        if pd.isna(pot) or np.isinf(pot):
            return 'Inconnu'
        if 0 <= pot < 3:
            return 'Très faible'
        elif 3 <= pot < 6:
            return 'Faible'
        elif 6 <= pot < 10:
            return 'Modéré'
        elif 10 <= pot < 15:
            return 'Élevé'
        elif pot >= 15:
            return 'Très élevé'
        return 'Inconnu'
    
    df['Potentiel_Categorie'] = df['Potentiel_Destructeur'].apply(categorize_potentiel)
    
    return df

def show_advanced_filters(df):
    """Afficher les filtres avancés pour l'analyse des caractéristiques"""
    
    with st.expander("🔧 Filtres Avancés par Catégories", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filtre par catégorie de magnitude
            mag_categories = ['Micro', 'Faible', 'Léger', 'Modéré', 'Fort', 'Majeur', 'Grand']
            available_mag_cats = [cat for cat in mag_categories if cat in df['Magnitude_Categorie'].unique()]
            
            selected_mag_cats = st.multiselect(
                "Catégories de magnitude",
                available_mag_cats,
                default=available_mag_cats,
                help="Filtrer par niveau de magnitude"
            )
        
        with col2:
            # Filtre par catégorie de profondeur
            depth_categories = ['Peu profond', 'Intermédiaire', 'Profond']
            available_depth_cats = [cat for cat in depth_categories if cat in df['Profondeur_Categorie'].unique()]
            
            selected_depth_cats = st.multiselect(
                "Catégories de profondeur",
                available_depth_cats,
                default=available_depth_cats,
                help="Filtrer par niveau de profondeur"
            )
        
        with col3:
            # Filtre par potentiel destructeur
            pot_categories = ['Très faible', 'Faible', 'Modéré', 'Élevé', 'Très élevé']
            available_pot_cats = [cat for cat in pot_categories if cat in df['Potentiel_Categorie'].unique()]
            
            selected_pot_cats = st.multiselect(
                "Potentiel destructeur",
                available_pot_cats,
                default=available_pot_cats,
                help="Filtrer par potentiel de destruction"
            )
        
        # Appliquer les filtres avancés
        if selected_mag_cats:
            df = df[df['Magnitude_Categorie'].isin(selected_mag_cats)]
        if selected_depth_cats:
            df = df[df['Profondeur_Categorie'].isin(selected_depth_cats)]
        if selected_pot_cats:
            df = df[df['Potentiel_Categorie'].isin(selected_pot_cats)]
        
        # Mettre à jour les données filtrées
        st.session_state.filtered_df = df
        
        if len(df) == 0:
            st.warning("⚠️ Aucune donnée ne correspond aux filtres avancés.")

def show_key_metrics(df):
    """Afficher les métriques clés des caractéristiques sismiques"""
    
    st.subheader("📊 Métriques Clés")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Magnitude moyenne", f"{df['Magnitude'].mean():.2f}")
        
    with col2:
        st.metric("Profondeur moyenne", f"{df['Profondeur'].mean():.1f} km")
        
    with col3:
        énergie_totale = df['Energie'].sum()
        st.metric("Énergie totale", f"{énergie_totale:.2e} J")
        
    with col4:
        potentiel_max = df['Potentiel_Destructeur'].max()
        st.metric("Potentiel max", f"{potentiel_max:.1f}")
        
    with col5:
        séismes_dangereux = len(df[df['Potentiel_Destructeur'] > 10])
        st.metric("Séismes élevés", séismes_dangereux)

def analyser_distribution_magnitudes(df_filtered):
    """Analyser la distribution des magnitudes"""
    
    st.subheader("📊 Analyse des Magnitudes")
    
    if len(df_filtered) == 0:
        st.warning("Aucune donnée pour l'analyse des magnitudes.")
        return
    
    # 1. Distribution globale
    st.markdown("#### 📈 Distribution globale des magnitudes")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogramme avec courbe de densité
    ax1.hist(df_filtered['Magnitude'], bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    
    # Ajouter une courbe de densité
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(df_filtered['Magnitude'])
    x_range = np.linspace(df_filtered['Magnitude'].min(), df_filtered['Magnitude'].max(), 100)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Densité estimée')
    
    ax1.axvline(df_filtered['Magnitude'].mean(), color='red', linestyle='--', 
               label=f'Moyenne: {df_filtered["Magnitude"].mean():.2f}')
    ax1.axvline(df_filtered['Magnitude'].median(), color='green', linestyle='--', 
               label=f'Médiane: {df_filtered["Magnitude"].median():.2f}')
    
    ax1.set_title('Distribution des magnitudes')
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('Densité')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot
    ax2.boxplot(df_filtered['Magnitude'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_title('Box plot des magnitudes')
    ax2.set_ylabel('Magnitude')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 2. Distribution par catégorie
    st.markdown("#### 🏷️ Distribution par catégorie")
    
    order = ['Micro', 'Faible', 'Léger', 'Modéré', 'Fort', 'Majeur', 'Grand']
    order = [cat for cat in order if cat in df_filtered['Magnitude_Categorie'].unique()]
    
    mag_counts = df_filtered['Magnitude_Categorie'].value_counts().reindex(order)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(order)))
    bars = ax.bar(order, mag_counts.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Ajouter les valeurs et pourcentages
    total = len(df_filtered)
    for i, (bar, count) in enumerate(zip(bars, mag_counts.values)):
        percentage = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(mag_counts.values) * 0.01,
               f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Nombre de séismes par catégorie de magnitude')
    ax.set_xlabel('Catégorie de magnitude')
    ax.set_ylabel('Nombre de séismes')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 3. Statistiques détaillées
    st.markdown("#### 📋 Statistiques détaillées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="stats-container">
            <h4>📊 Statistiques descriptives</h4>
        </div>
        """, unsafe_allow_html=True)
        
        stats_data = {
            "Statistique": ["Nombre total", "Minimum", "Maximum", "Moyenne", "Médiane", "Écart-type", "Skewness", "Kurtosis"],
            "Valeur": [
                len(df_filtered),
                f"{df_filtered['Magnitude'].min():.2f}",
                f"{df_filtered['Magnitude'].max():.2f}",
                f"{df_filtered['Magnitude'].mean():.2f}",
                f"{df_filtered['Magnitude'].median():.2f}",
                f"{df_filtered['Magnitude'].std():.2f}",
                f"{stats.skew(df_filtered['Magnitude']):.2f}",
                f"{stats.kurtosis(df_filtered['Magnitude']):.2f}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True)
    
    with col2:
        st.markdown("""
        <div class="stats-container">
            <h4>🏷️ Répartition par catégorie</h4>
        </div>
        """, unsafe_allow_html=True)
        
        category_data = []
        for category in order:
            count = mag_counts[category] if category in mag_counts.index else 0
            percentage = count / total * 100 if total > 0 else 0
            category_data.append({
                "Catégorie": category,
                "Nombre": count,
                "Pourcentage": f"{percentage:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(category_data), hide_index=True)

def analyser_distribution_profondeurs(df_filtered):
    """Analyser la distribution des profondeurs"""
    
    st.subheader("🕳️ Analyse des Profondeurs")
    
    if len(df_filtered) == 0:
        st.warning("Aucune donnée pour l'analyse des profondeurs.")
        return
    
    # 1. Distribution globale
    st.markdown("#### 📈 Distribution globale des profondeurs")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogramme avec courbe de densité
    ax1.hist(df_filtered['Profondeur'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
    
    # Ajouter une courbe de densité
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(df_filtered['Profondeur'])
    x_range = np.linspace(df_filtered['Profondeur'].min(), df_filtered['Profondeur'].max(), 100)
    ax1.plot(x_range, kde(x_range), 'darkred', linewidth=2, label='Densité estimée')
    
    ax1.axvline(df_filtered['Profondeur'].mean(), color='red', linestyle='--', 
               label=f'Moyenne: {df_filtered["Profondeur"].mean():.1f} km')
    ax1.axvline(df_filtered['Profondeur'].median(), color='green', linestyle='--', 
               label=f'Médiane: {df_filtered["Profondeur"].median():.1f} km')
    
    ax1.set_title('Distribution des profondeurs')
    ax1.set_xlabel('Profondeur (km)')
    ax1.set_ylabel('Densité')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Échelle logarithmique pour mieux voir la distribution
    ax2.hist(df_filtered['Profondeur'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_title('Distribution des profondeurs (échelle log)')
    ax2.set_xlabel('Profondeur (km)')
    ax2.set_ylabel('Nombre de séismes (log)')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 2. Distribution par catégorie
    st.markdown("#### 🏷️ Distribution par catégorie de profondeur")
    
    order = ['Peu profond', 'Intermédiaire', 'Profond']
    order = [cat for cat in order if cat in df_filtered['Profondeur_Categorie'].unique()]
    
    depth_counts = df_filtered['Profondeur_Categorie'].value_counts().reindex(order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['lightblue', 'steelblue', 'darkblue'][:len(order)]
    bars = ax.bar(order, depth_counts.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Ajouter les valeurs et pourcentages
    total = len(df_filtered)
    for i, (bar, count) in enumerate(zip(bars, depth_counts.values)):
        percentage = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(depth_counts.values) * 0.01,
               f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Nombre de séismes par catégorie de profondeur')
    ax.set_xlabel('Catégorie de profondeur')
    ax.set_ylabel('Nombre de séismes')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 3. Impact de la profondeur sur les dégâts potentiels
    st.markdown("#### ⚠️ Impact de la profondeur sur les dégâts")
    
    st.markdown("""
    <div class="danger-alert">
        <h4>🚨 Relation profondeur-dégâts</h4>
        <p><strong>Séismes peu profonds (< 70 km) :</strong> Plus destructeurs en surface</p>
        <p><strong>Séismes intermédiaires (70-300 km) :</strong> Impact modéré</p>
        <p><strong>Séismes profonds (> 300 km) :</strong> Moins ressentis en surface</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Analyse par catégorie avec magnitude moyenne
    fig, ax = plt.subplots(figsize=(12, 6))
    
    avg_magnitude_by_depth = df_filtered.groupby('Profondeur_Categorie')['Magnitude'].mean().reindex(order)
    
    bars = ax.bar(order, avg_magnitude_by_depth.values, color=colors, alpha=0.8, edgecolor='black')
    
    for i, (bar, mag) in enumerate(zip(bars, avg_magnitude_by_depth.values)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
               f'{mag:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Magnitude moyenne par catégorie de profondeur')
    ax.set_xlabel('Catégorie de profondeur')
    ax.set_ylabel('Magnitude moyenne')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def analyser_relation_magnitude_profondeur(df_filtered):
    """Analyser la relation entre magnitude et profondeur"""
    
    st.subheader("🔗 Relation Magnitude-Profondeur")
    
    if len(df_filtered) == 0:
        st.warning("Aucune donnée pour l'analyse des relations.")
        return
    
    # 1. Nuage de points avec potentiel destructeur
    st.markdown("#### 🎯 Nuage de points avec potentiel destructeur")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(df_filtered['Profondeur'], df_filtered['Magnitude'], 
                        c=df_filtered['Potentiel_Destructeur'], cmap='YlOrRd', 
                        alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    plt.colorbar(scatter, label='Potentiel destructeur')
    
    # Ajouter une régression linéaire
    if len(df_filtered) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_filtered['Profondeur'], df_filtered['Magnitude'])
        
        x_line = np.array([df_filtered['Profondeur'].min(), df_filtered['Profondeur'].max()])
        y_line = intercept + slope * x_line
        
        ax.plot(x_line, y_line, 'b--', linewidth=2,
               label=f'Régression: y={slope:.4f}x+{intercept:.2f} (r²={r_value**2:.3f})')
        
        # Analyse statistique
        significance = "significative" if p_value < 0.05 else "non significative"
        direction = "positive" if slope > 0 else "négative"
        
        st.markdown(f"""
        <div class="stats-container">
            <h4>📊 Analyse de corrélation</h4>
            <p><strong>Coefficient de corrélation (r) :</strong> {r_value:.3f}</p>
            <p><strong>Coefficient de détermination (r²) :</strong> {r_value**2:.3f}</p>
            <p><strong>p-value :</strong> {p_value:.4f}</p>
            <p><strong>Conclusion :</strong> Corrélation {direction} {significance}</p>
        </div>
        """, unsafe_allow_html=True)
    
    ax.set_title('Relation entre magnitude et profondeur\n(couleur = potentiel destructeur)')
    ax.set_xlabel('Profondeur (km)')
    ax.set_ylabel('Magnitude')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def analyser_potentiel_destructeur(df_filtered):
    """Analyser le potentiel destructeur - VERSION CORRIGÉE"""
    
    st.subheader("⚠️ Analyse du Potentiel Destructeur")
    
    if len(df_filtered) == 0:
        st.warning("Aucune donnée pour l'analyse du potentiel destructeur.")
        return
    
    # CORRECTION : Vérifications supplémentaires avant l'analyse
    if 'Potentiel_Destructeur' not in df_filtered.columns:
        st.error("❌ Colonne 'Potentiel_Destructeur' manquante.")
        return
    
    # Nettoyer les données avant l'analyse
    df_clean = df_filtered.copy()
    
    # Supprimer les valeurs NaN ou infinies
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['Potentiel_Destructeur'].notna()]
    df_clean = df_clean[~np.isinf(df_clean['Potentiel_Destructeur'])]
    df_clean = df_clean[df_clean['Potentiel_Destructeur'] >= 0]  # Valeurs positives seulement
    
    cleaned_count = len(df_clean)
    if cleaned_count < initial_count:
        st.info(f"ℹ️ {initial_count - cleaned_count} valeurs invalides supprimées de l'analyse.")
    
    if len(df_clean) == 0:
        st.error("❌ Aucune donnée valide pour l'analyse du potentiel destructeur.")
        return
    
    # 1. Distribution du potentiel destructeur
    st.markdown("#### 📊 Distribution du potentiel destructeur")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogramme - avec vérification des données
    try:
        potentiel_data = df_clean['Potentiel_Destructeur'].values
        
        # Vérifier qu'il y a des données et qu'elles sont valides
        if len(potentiel_data) > 0 and not np.all(np.isnan(potentiel_data)):
            ax1.hist(potentiel_data, bins=min(30, len(potentiel_data)//2), alpha=0.7, 
                     color='orange', edgecolor='black', density=True)
            
            mean_val = np.nanmean(potentiel_data)
            median_val = np.nanmedian(potentiel_data)
            
            ax1.axvline(mean_val, color='red', linestyle='--', 
                       label=f'Moyenne: {mean_val:.2f}')
            ax1.axvline(median_val, color='green', linestyle='--', 
                       label=f'Médiane: {median_val:.2f}')
            
            ax1.set_title('Distribution du potentiel destructeur')
            ax1.set_xlabel('Potentiel destructeur')
            ax1.set_ylabel('Densité')
            ax1.legend()
            ax1.grid(alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Données insuffisantes', transform=ax1.transAxes, 
                    ha='center', va='center', fontsize=14)
            ax1.set_title('Distribution du potentiel destructeur - Données insuffisantes')
    
    except Exception as e:
        st.error(f"Erreur lors de la création de l'histogramme: {str(e)}")
        ax1.text(0.5, 0.5, 'Erreur de visualisation', transform=ax1.transAxes, 
                ha='center', va='center', fontsize=14)
    
    # Distribution par catégorie
    try:
        order = ['Très faible', 'Faible', 'Modéré', 'Élevé', 'Très élevé']
        order = [cat for cat in order if cat in df_clean['Potentiel_Categorie'].unique()]
        
        if len(order) > 0:
            potentiel_counts = df_clean['Potentiel_Categorie'].value_counts().reindex(order)
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(order)))
            
            bars = ax2.bar(order, potentiel_counts.values, color=colors, alpha=0.8, edgecolor='black')
            
            total = len(df_clean)
            for i, (bar, count) in enumerate(zip(bars, potentiel_counts.values)):
                if not pd.isna(count) and count > 0:
                    percentage = count / total * 100
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{int(count)}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_title('Nombre de séismes par catégorie de potentiel')
            ax2.set_xlabel('Catégorie de potentiel destructeur')
            ax2.set_ylabel('Nombre de séismes')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Aucune catégorie disponible', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=14)
    
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique par catégorie: {str(e)}")
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 2. Analyse des séismes les plus dangereux
    st.markdown("#### 🚨 Séismes à fort potentiel destructeur")
    
    try:
        # Identifier les séismes les plus dangereux (top 10%)
        if len(df_clean) > 10:  # S'assurer qu'il y a assez de données
            seuil_danger = df_clean['Potentiel_Destructeur'].quantile(0.9)
            seismes_dangereux = df_clean[df_clean['Potentiel_Destructeur'] >= seuil_danger]
            
            if len(seismes_dangereux) > 0:
                st.markdown(f"""
                <div class="danger-alert">
                    <h4>⚠️ Séismes à surveiller</h4>
                    <p><strong>{len(seismes_dangereux)} séismes</strong> ont un potentiel destructeur élevé (≥ {seuil_danger:.1f})</p>
                    <p>Ces séismes représentent <strong>{len(seismes_dangereux)/len(df_clean)*100:.1f}%</strong> de l'ensemble</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tableau des séismes les plus dangereux
                if 'Date' in seismes_dangereux.columns:
                    top_dangerous = seismes_dangereux.nlargest(10, 'Potentiel_Destructeur')[
                        ['Date', 'Magnitude', 'Profondeur', 'Potentiel_Destructeur']
                    ].copy()
                    
                    # Conversion sécurisée des dates
                    try:
                        # Essayer différents formats de date
                        top_dangerous['Date'] = pd.to_datetime(top_dangerous['Date'], format='%d/%m/%y %H:%M', errors='coerce')
                        if top_dangerous['Date'].isna().all():
                            # Si le premier format échoue, essayer avec année complète
                            top_dangerous['Date'] = pd.to_datetime(top_dangerous['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
                        if top_dangerous['Date'].isna().all():
                            # Si les formats échouent, utiliser l'inférence automatique
                            top_dangerous['Date'] = pd.to_datetime(top_dangerous['Date'], errors='coerce')
                        
                        # Formater les dates pour l'affichage
                        top_dangerous['Date'] = top_dangerous['Date'].dt.strftime('%d/%m/%Y %H:%M')
                    except:
                        # En cas d'erreur, garder les dates comme chaînes
                        pass
                    
                    top_dangerous.columns = ['Date', 'Magnitude', 'Profondeur (km)', 'Potentiel']
                    st.dataframe(top_dangerous, hide_index=True, use_container_width=True)
                else:
                    # Si pas de colonne Date, afficher sans
                    top_dangerous = seismes_dangereux.nlargest(10, 'Potentiel_Destructeur')[
                        ['Magnitude', 'Profondeur', 'Potentiel_Destructeur']
                    ].copy()
                    top_dangerous.columns = ['Magnitude', 'Profondeur (km)', 'Potentiel']
                    st.dataframe(top_dangerous, hide_index=True, use_container_width=True)
            else:
                st.info("ℹ️ Aucun séisme avec un potentiel destructeur particulièrement élevé détecté.")
        else:
            st.info("ℹ️ Données insuffisantes pour l'analyse des séismes dangereux.")
    
    except Exception as e:
        st.error(f"Erreur lors de l'analyse des séismes dangereux: {str(e)}")
        
    # 3. Statistiques du potentiel destructeur
    st.markdown("#### 📊 Statistiques du potentiel destructeur")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="stats-container">
                <h4>📊 Statistiques descriptives</h4>
            </div>
            """, unsafe_allow_html=True)
            
            stats_data = {
                "Statistique": ["Nombre total", "Minimum", "Maximum", "Moyenne", "Médiane", "Écart-type"],
                "Valeur": [
                    len(df_clean),
                    f"{df_clean['Potentiel_Destructeur'].min():.2f}",
                    f"{df_clean['Potentiel_Destructeur'].max():.2f}",
                    f"{df_clean['Potentiel_Destructeur'].mean():.2f}",
                    f"{df_clean['Potentiel_Destructeur'].median():.2f}",
                    f"{df_clean['Potentiel_Destructeur'].std():.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), hide_index=True)
        
        with col2:
            if len(order) > 0:
                st.markdown("""
                <div class="stats-container">
                    <h4>🏷️ Répartition par catégorie</h4>
                </div>
                """, unsafe_allow_html=True)
                
                category_data = []
                for category in order:
                    count = potentiel_counts[category] if category in potentiel_counts.index else 0
                    percentage = count / len(df_clean) * 100 if len(df_clean) > 0 else 0
                    category_data.append({
                        "Catégorie": category,
                        "Nombre": int(count) if not pd.isna(count) else 0,
                        "Pourcentage": f"{percentage:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(category_data), hide_index=True)
    
    except Exception as e:
        st.error(f"Erreur lors du calcul des statistiques: {str(e)}")

def analyser_energie(df_filtered):
    """Analyser l'énergie libérée par les séismes"""
    
    st.subheader("⚡ Analyse de l'Énergie Libérée")
    
    if len(df_filtered) == 0:
        st.warning("Aucune donnée pour l'analyse de l'énergie.")
        return
    
    # 1. Distribution de l'énergie
    st.markdown("#### 📊 Distribution de l'énergie libérée")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution normale
    ax1.hist(df_filtered['Energie'], bins=30, alpha=0.7, color='gold', edgecolor='black')
    ax1.set_title('Distribution de l\'énergie')
    ax1.set_xlabel('Énergie (Joules)')
    ax1.set_ylabel('Nombre de séismes')
    ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    ax1.grid(alpha=0.3)
    
    # Distribution logarithmique
    log_energie = np.log10(df_filtered['Energie'])
    ax2.hist(log_energie, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_title('Distribution de l\'énergie (échelle log)')
    ax2.set_xlabel('Énergie (log₁₀ Joules)')
    ax2.set_ylabel('Nombre de séismes')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 2. Énergie cumulée dans le temps
    st.markdown("#### 📈 Évolution de l'énergie cumulée")
    
    if 'Date' in df_filtered.columns:
        try:
            # Conversion sécurisée des dates
            df_sorted = df_filtered.copy()
            
            # Essayer différents formats de date
            try:
                df_sorted['Date_converted'] = pd.to_datetime(df_sorted['Date'], format='%d/%m/%y %H:%M', errors='coerce')
            except:
                try:
                    df_sorted['Date_converted'] = pd.to_datetime(df_sorted['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
                except:
                    df_sorted['Date_converted'] = pd.to_datetime(df_sorted['Date'], errors='coerce')
            
            # Vérifier si la conversion a réussi
            if df_sorted['Date_converted'].notna().any():
                df_sorted = df_sorted[df_sorted['Date_converted'].notna()].sort_values('Date_converted')
                energie_cumulee = df_sorted['Energie'].cumsum()
                
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(df_sorted['Date_converted'], energie_cumulee, linewidth=2, color='red')
                ax.set_title('Énergie sismique cumulée au fil du temps')
                ax.set_xlabel('Date')
                ax.set_ylabel('Énergie cumulée (Joules)')
                ax.set_yscale('log')
                ax.grid(alpha=0.3)
                
                # Ajouter des informations sur les pics d'énergie
                try:
                    max_daily_energy = df_sorted.groupby(df_sorted['Date_converted'].dt.date)['Energie'].sum()
                    if len(max_daily_energy) > 0:
                        top_energy_day = max_daily_energy.idxmax()
                        max_energy_value = max_daily_energy.max()
                        
                        st.markdown(f"""
                        <div class="energy-metric">
                            <h4>⚡ Pic d'énergie</h4>
                            <p><strong>Jour le plus énergétique :</strong> {top_energy_day}</p>
                            <p><strong>Énergie libérée :</strong> {max_energy_value:.2e} Joules</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.info(f"ℹ️ Impossible de calculer les pics d'énergie: {str(e)}")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("⚠️ Impossible de convertir les dates pour l'analyse temporelle.")
        
        except Exception as e:
            st.warning(f"⚠️ Erreur lors de l'analyse temporelle: {str(e)}")
            st.info("ℹ️ L'analyse temporelle a été ignorée en raison du format des dates.")
    
    # 3. Répartition de l'énergie par catégorie de magnitude
    st.markdown("#### 🏷️ Répartition de l'énergie par catégorie")
    
    order = ['Micro', 'Faible', 'Léger', 'Modéré', 'Fort', 'Majeur', 'Grand']
    order = [cat for cat in order if cat in df_filtered['Magnitude_Categorie'].unique()]
    
    # Calculer l'énergie totale par catégorie
    energie_par_categorie = df_filtered.groupby('Magnitude_Categorie')['Energie'].sum().reindex(order)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(order)))
    bars = ax.bar(order, energie_par_categorie.values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('Énergie totale libérée par catégorie de magnitude')
    ax.set_xlabel('Catégorie de magnitude')
    ax.set_ylabel('Énergie totale (Joules)')
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3)
    
    # Ajouter les pourcentages
    total_energy = energie_par_categorie.sum()
    for i, (bar, energy) in enumerate(zip(bars, energie_par_categorie.values)):
        percentage = energy / total_energy * 100
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
               f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 4. Relation énergie-magnitude (loi de Gutenberg-Richter)
    st.markdown("#### 📏 Relation énergie-magnitude")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scatter = ax.scatter(df_filtered['Magnitude'], np.log10(df_filtered['Energie']), 
                        alpha=0.6, c=df_filtered['Profondeur'], cmap='viridis', s=50)
    
    plt.colorbar(scatter, label='Profondeur (km)')
    
    # Ajouter la relation théorique E = 10^(1.5*M+4.8)
    mag_theory = np.linspace(df_filtered['Magnitude'].min(), df_filtered['Magnitude'].max(), 100)
    log_energy_theory = 1.5 * mag_theory + 4.8
    
    ax.plot(mag_theory, log_energy_theory, 'r--', linewidth=2, 
           label='Relation théorique: log₁₀(E) = 1.5M + 4.8')
    
    ax.set_title('Relation entre magnitude et énergie libérée')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Énergie (log₁₀ Joules)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 5. Statistiques énergétiques
    st.markdown("#### 📊 Statistiques énergétiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="energy-metric">
            <h4>⚡ Métriques globales</h4>
        </div>
        """, unsafe_allow_html=True)
        
        total_energie = df_filtered['Energie'].sum()
        moyenne_energie = df_filtered['Energie'].mean()
        max_energie = df_filtered['Energie'].max()
        
        energy_stats = {
            "Métrique": ["Énergie totale", "Énergie moyenne", "Énergie maximale", "Énergie médiane"],
            "Valeur": [
                f"{total_energie:.2e} J",
                f"{moyenne_energie:.2e} J",
                f"{max_energie:.2e} J",
                f"{df_filtered['Energie'].median():.2e} J"
            ]
        }
        st.dataframe(pd.DataFrame(energy_stats), hide_index=True)
    
    with col2:
        st.markdown("""
        <div class="energy-metric">
            <h4>🏷️ Contribution par catégorie</h4>
        </div>
        """, unsafe_allow_html=True)
        
        energy_contribution = []
        for category in order:
            if category in df_filtered['Magnitude_Categorie'].unique():
                cat_energy = df_filtered[df_filtered['Magnitude_Categorie'] == category]['Energie'].sum()
                percentage = cat_energy / total_energie * 100
                count = len(df_filtered[df_filtered['Magnitude_Categorie'] == category])
                
                energy_contribution.append({
                    "Catégorie": category,
                    "Énergie": f"{cat_energy:.2e} J",
                    "Contribution": f"{percentage:.1f}%",
                    "Nombre": count
                })
        
        if energy_contribution:
            st.dataframe(pd.DataFrame(energy_contribution), hide_index=True)

# Fonction principale qui peut être appelée depuis app.py
def main():
    """Fonction principale à appeler depuis l'application principale"""
    show_analyse_caracteristiques()

if __name__ == "__main__":
    main()
