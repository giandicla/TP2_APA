import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
from typing import Optional, Sequence
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize

# Configuración de la página
st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎵",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #b3b3b3;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1DB954;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1ed760;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado
st.markdown('<h1 class="main-header">🎵 Spotify Song Recommender</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Busca una canción y obtén recomendaciones basadas en sus características</p>', unsafe_allow_html=True)


@st.cache_resource
def load_and_train_model():
    """Carga los datos y entrena el modelo completo"""
    
    with st.spinner("🔄 Cargando datos y entrenando modelo..."):
        # ---------------------------
        # 0) Cargar datos
        # ---------------------------
        df = pd.read_csv('data.csv')
        
        # ---------------------------
        # 1) Limpieza de artistas
        # ---------------------------
        def _ensure_artist_sequence(value):
            if isinstance(value, (list, tuple)):
                return [v for v in value if isinstance(v, str)]
            if isinstance(value, str):
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, (list, tuple)):
                        return [v for v in parsed if isinstance(v, str)]
                except Exception:
                    pass
            return []

        if 'artists' not in df.columns:
            df['artists'] = [[] for _ in range(len(df))]
        else:
            df['artists'] = df['artists'].apply(_ensure_artist_sequence)

        def artists_to_str(lst: Sequence[str]) -> str:
            tokens = []
            for artist in lst:
                clean = re.sub(r"\s+", '_', artist.strip().lower())
                if clean:
                    tokens.append(clean)
            return ' '.join(tokens)

        df['artist_primary'] = df['artists'].apply(lambda lst: lst[0] if lst else None)
        df['artists_str'] = df['artists'].apply(artists_to_str)

        # ---------------------------
        # 2) Feature engineering
        # ---------------------------
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['decade'] = (df['year'] // 10) * 10
            current_year = pd.Timestamp.now().year
            df['years_since_release'] = current_year - df['year']
            df['year/recency'] = df['year'] / (current_year - df['year'] + 1)
        else:
            df['decade'] = 0
            df['years_since_release'] = 0
            df['year/recency'] = 0

        if 'release_date' in df.columns:
            df['release_date_parsed'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['release_month'] = df['release_date_parsed'].dt.month.fillna(0).astype(int)
            df['release_dayofyear'] = df['release_date_parsed'].dt.dayofyear.fillna(0).astype(int)
            df['release_month_sin'] = np.sin(2 * np.pi * (df['release_month'] / 12.0))
            df['release_month_cos'] = np.cos(2 * np.pi * (df['release_month'] / 12.0))
            df['release_doy_sin'] = np.sin(2 * np.pi * (df['release_dayofyear'] / 365.0))
            df['release_doy_cos'] = np.cos(2 * np.pi * (df['release_dayofyear'] / 365.0))
        else:
            df['release_month_sin'] = 0
            df['release_month_cos'] = 0
            df['release_doy_sin'] = 0
            df['release_doy_cos'] = 0

        cof_order = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: 7, 8: 8, 3: 9, 10: 10, 5: 11}
        df['key'] = pd.to_numeric(df['key'], errors='coerce') if 'key' in df.columns else np.nan
        df['key_cof_pos'] = df['key'].map(cof_order)
        mask_cof = df['key_cof_pos'].notna()
        df.loc[mask_cof, 'key_cof_sin'] = np.sin(2 * np.pi * df.loc[mask_cof, 'key_cof_pos'] / 12)
        df.loc[mask_cof, 'key_cof_cos'] = np.cos(2 * np.pi * df.loc[mask_cof, 'key_cof_pos'] / 12)
        df['key_cof_sin'] = df['key_cof_sin'].fillna(0)
        df['key_cof_cos'] = df['key_cof_cos'].fillna(0)

        if 'mode' in df.columns:
            df['mode'] = df['mode'].fillna(0).astype(int)
            mask_k = df['key'].notna()
            df.loc[mask_k, 'key_mode_idx'] = df.loc[mask_k, 'key'].astype(int) + 12 * df.loc[mask_k, 'mode']
            df.loc[mask_k, 'key24_sin'] = np.sin(2 * np.pi * df.loc[mask_k, 'key_mode_idx'] / 24)
            df.loc[mask_k, 'key24_cos'] = np.cos(2 * np.pi * df.loc[mask_k, 'key_mode_idx'] / 24)
            df['key24_sin'] = df['key24_sin'].fillna(0)
            df['key24_cos'] = df['key24_cos'].fillna(0)
        else:
            df['key24_sin'] = 0
            df['key24_cos'] = 0

        for col in ['popularity','valence','danceability','energy','acousticness','duration_ms','speechiness','tempo','loudness','instrumentalness','liveness']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['explicit_flag'] = df.get('explicit', 0).fillna(0).astype(int)
        df['duration_minutes'] = df.get('duration_ms', 0).fillna(0) / 60000.0
        df['energy_danceability'] = df.get('energy', 0).fillna(0) * df.get('danceability', 0).fillna(0)
        df['acoustic_energy_ratio'] = df.get('acousticness', 0).fillna(0) / (df.get('energy', 0).fillna(0) + 1e-3)
        df['valence_energy_gap'] = (df.get('valence', 0).fillna(0) - df.get('energy', 0).fillna(0)).abs()
        df['popularity_rank'] = df['popularity'].rank(pct=True).fillna(0) if 'popularity' in df else 0

        # ---------------------------
        # 3) Estadísticas por artista
        # ---------------------------
        if df['artist_primary'].notna().any():
            artist_stats = (
                df[df['artist_primary'].notna()]
                .groupby('artist_primary')
                .agg(
                    artist_pop_mean=('popularity', 'mean'),
                    artist_pop_std=('popularity', 'std'),
                    artist_energy_mean=('energy', 'mean'),
                    artist_dance_mean=('danceability', 'mean'),
                    artist_valence_mean=('valence', 'mean'),
                    artist_year_mean=('year', 'mean'),
                    artist_track_count=('id', 'count')
                )
            )
            df = df.merge(artist_stats, how='left', left_on='artist_primary', right_index=True)
        else:
            for col in ['artist_pop_mean','artist_pop_std','artist_energy_mean','artist_dance_mean','artist_valence_mean','artist_year_mean','artist_track_count']:
                df[col] = 0

        for col in ['artist_pop_mean','artist_pop_std','artist_energy_mean','artist_dance_mean','artist_valence_mean','artist_year_mean','artist_track_count']:
            df[col] = df[col].fillna(0)

        df['artist_track_count'] = df['artist_track_count'].replace(0, 1)
        df['artist_track_count_log'] = np.log1p(df['artist_track_count'])
        df['artist_recency_gap'] = (df['year'] - df['artist_year_mean']).fillna(0)
        df['artist_pop_delta'] = (df['popularity'] - df['artist_pop_mean']).fillna(0)

        # ---------------------------
        # 4) Numeric features -> PCA
        # ---------------------------
        numeric_feats = [
            'valence','acousticness','danceability','duration_ms','duration_minutes','energy','energy_danceability',
            'instrumentalness','liveness','loudness','speechiness','tempo','popularity','popularity_rank',
            'decade','years_since_release','year/recency','artist_recency_gap','artist_year_mean',
            'key_cof_sin','key_cof_cos','key24_sin','key24_cos',
            'release_month_sin','release_month_cos','release_doy_sin','release_doy_cos',
            'explicit_flag','acoustic_energy_ratio','valence_energy_gap',
            'artist_pop_mean','artist_pop_std','artist_pop_delta','artist_energy_mean','artist_dance_mean','artist_valence_mean',
            'artist_track_count_log'
        ]

        for feat in list(numeric_feats):
            if feat not in df.columns:
                df[feat] = 0

        X_num = df[numeric_feats].fillna(0).values
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        n_num_components = min(24, X_num_scaled.shape[1])
        pca_num = PCA(n_components=n_num_components, random_state=42)
        X_num_emb = pca_num.fit_transform(X_num_scaled)

        # ---------------------------
        # 5) Text embedding
        # ---------------------------
        df['text_for_tfidf'] = (
            df.get('name', '').fillna('').astype(str) + ' ' +
            df['artists_str'].fillna('') + ' ' +
            df['decade'].fillna(0).astype(int).astype(str)
        )
        tfidf_text = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2, token_pattern=r'(?u)\b\w+\b')
        X_text_tfidf = tfidf_text.fit_transform(df['text_for_tfidf'].fillna(''))
        n_text_comp = min(96, X_text_tfidf.shape[1] - 1 if X_text_tfidf.shape[1] > 1 else 1)
        svd_text = TruncatedSVD(n_components=n_text_comp, random_state=42)
        X_text_emb = svd_text.fit_transform(X_text_tfidf)

        # ---------------------------
        # 6) Artist embedding
        # ---------------------------
        artists_str_filled = df['artists_str'].fillna('')
        X_art_emb = None
        if (artists_str_filled.astype(str).str.strip() != '').any():
            try:
                tfidf_art = TfidfVectorizer(max_features=800, token_pattern=r'(?u)\b\w+\b')
                X_art_tfidf = tfidf_art.fit_transform(artists_str_filled)
                if X_art_tfidf.shape[1] > 0:
                    n_art = min(24, X_art_tfidf.shape[1] - 1 if X_art_tfidf.shape[1] > 1 else 1)
                    svd_art = TruncatedSVD(n_components=n_art, random_state=42)
                    X_art_emb = svd_art.fit_transform(X_art_tfidf)
            except Exception:
                X_art_emb = None

        if X_art_emb is None:
            try:
                mlb = MultiLabelBinarizer()
                artists_lists = df['artists']
                X_mlb = mlb.fit_transform(artists_lists)
                if X_mlb.shape[1] == 0:
                    X_art_emb = np.zeros((len(df), 1))
                else:
                    n_art = min(16, X_mlb.shape[1] - 1 if X_mlb.shape[1] > 1 else 1)
                    svd_art = TruncatedSVD(n_components=n_art, random_state=42)
                    X_art_emb = svd_art.fit_transform(X_mlb)
            except Exception:
                X_art_emb = np.zeros((len(df), 1))

        # ---------------------------
        # 7) Normalize and combine
        # ---------------------------
        X_num_n = normalize(X_num_emb, axis=1)
        X_text_n = normalize(X_text_emb, axis=1)
        X_art_n = normalize(X_art_emb, axis=1)

        num_w, text_w, art_w = 1.2, 0.9, 1.4
        X_comb = np.hstack([X_num_n * num_w, X_text_n * text_w, X_art_n * art_w])

        # ---------------------------
        # 8) PCA final
        # ---------------------------
        final_dim = min(80, X_comb.shape[1])
        pca_final = PCA(n_components=final_dim, random_state=42)
        X_final = pca_final.fit_transform(X_comb)
        X_final_norm = normalize(X_final, norm='l2')

        # ---------------------------
        # 9) Train NN
        # ---------------------------
        n_neighbors = min(len(df), 50)
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn.fit(X_final_norm)

        return df, nn, X_final_norm


def recommend_by_track_id(
    df, nn, X_final_norm, track_id: str,
    n_recs: int = 10,
    rerank_by_artist: bool = False,
    pop_weight: float = 0.0,
    boost_recency: float = 0.0,
    filter_explicit: Optional[int] = None,
    year_range: Optional[Sequence[int]] = None,
    min_popularity: Optional[float] = None,
):
    """Función de recomendación"""
    idx_list = df.index[df['id'] == track_id].tolist()
    if not idx_list:
        raise ValueError('Track ID no encontrado')
    idx = idx_list[0]
    
    fetch = min(len(df), max(5 * n_recs, n_recs + 25))
    vec = X_final_norm[idx].reshape(1, -1)
    distances, indices = nn.kneighbors(vec, n_neighbors=fetch)
    indices = indices[0]
    distances = distances[0]
    mask_self = indices != idx
    indices = indices[mask_self]
    distances = distances[mask_self]

    cols = ['id','name','artists','artist_primary','year','decade','popularity','explicit_flag','years_since_release','year/recency']
    recs = df.iloc[indices][cols].copy()
    recs['similarity'] = 1 - distances[: len(recs)]

    mask = pd.Series(True, index=recs.index)
    if filter_explicit is not None:
        mask &= recs['explicit_flag'] == int(filter_explicit)
    if year_range is not None:
        year_min, year_max = year_range
        mask &= recs['year'].between(year_min, year_max)
    if min_popularity is not None:
        mask &= recs['popularity'] >= min_popularity

    recs = recs[mask]
    if recs.empty:
        return recs

    score = recs['similarity'].copy()
    if rerank_by_artist:
        seed_artist = df.loc[idx, 'artist_primary']
        score += 0.1 * (recs['artist_primary'] == seed_artist).astype(float)
    if pop_weight:
        pop_norm = (recs['popularity'] - df['popularity'].min()) / (df['popularity'].max() - df['popularity'].min() + 1e-9)
        score += pop_weight * pop_norm
    if boost_recency:
        rec_norm = (recs['year/recency'] - recs['year/recency'].min()) / (recs['year/recency'].max() - recs['year/recency'].min() + 1e-9)
        score += boost_recency * rec_norm

    recs['score'] = score
    recs = recs.sort_values('score', ascending=False).head(n_recs).reset_index(drop=True)
    cols_out = ['id','name','artists','artist_primary','year','popularity','explicit_flag','similarity','score']
    return recs[cols_out]


# Cargar modelo
try:
    df, nn, X_final_norm = load_and_train_model()
    st.success("✅ Modelo cargado y entrenado exitosamente")
    
    # Sidebar con opciones avanzadas
    with st.sidebar:
        st.header("⚙️ Opciones de Recomendación")
        
        n_recs = st.slider("Cantidad de recomendaciones", min_value=5, max_value=30, value=10, step=1)
        
        st.subheader("Filtros")
        rerank_by_artist = st.checkbox("Priorizar mismo artista", value=False)
        pop_weight = st.slider("Peso de popularidad", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        boost_recency = st.slider("Preferir canciones recientes", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        
        filter_explicit = st.selectbox("Contenido explícito", options=[None, 0, 1], format_func=lambda x: "Sin filtro" if x is None else ("Solo limpio" if x == 0 else "Solo explícito"))
        
        use_year_filter = st.checkbox("Filtrar por año")
        if use_year_filter:
            year_min = int(df['year'].min()) if 'year' in df.columns else 1950
            year_max = int(df['year'].max()) if 'year' in df.columns else 2025
            year_range = st.slider("Rango de años", min_value=year_min, max_value=year_max, value=(year_min, year_max))
        else:
            year_range = None
        
        use_pop_filter = st.checkbox("Filtrar por popularidad mínima")
        if use_pop_filter:
            min_popularity = st.slider("Popularidad mínima", min_value=0, max_value=100, value=30)
        else:
            min_popularity = None

    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        cancion = st.text_input(
            "🔍 Nombre de la canción",
            placeholder="Ej: Bohemian Rhapsody, Dakiti, Watermelon Sugar...",
            help="No necesitas escribir el nombre exacto, se buscará por coincidencia parcial"
        )
    
    with col2:
        st.write("")
        st.write("")
        search_button = st.button("🎯 Buscar Recomendaciones")

    # Buscar y mostrar recomendaciones
    if search_button or (cancion and len(cancion) > 2):
        if cancion:
            try:
                # Buscar la canción
                matches = df[df['name'].str.contains(cancion, case=False, na=False)]
                
                if matches.empty:
                    st.warning(f"⚠️ No se encontró ninguna canción que contenga '{cancion}'")
                else:
                    # Mostrar coincidencias encontradas
                    st.subheader("📝 Canción encontrada")
                    
                    # Mostrar top 3 coincidencias
                    top_matches = matches.head(3)[['name', 'artists', 'year', 'popularity']]
                    
                    # Selector de canción si hay múltiples coincidencias
                    if len(matches) > 1:
                        selected_idx = st.selectbox(
                            "Se encontraron múltiples coincidencias. Selecciona una:",
                            range(len(top_matches)),
                            format_func=lambda i: f"{top_matches.iloc[i]['name']} - {top_matches.iloc[i]['artists']} ({top_matches.iloc[i]['year']})"
                        )
                        selected_song = matches.iloc[selected_idx]
                    else:
                        selected_song = matches.iloc[0]
                        st.success(f"✅ {selected_song['name']} - {selected_song['artists']} ({selected_song['year']})")
                    
                    # Mostrar características de la canción seleccionada
                    with st.expander("🎼 Ver características de la canción seleccionada"):
                        char_cols = st.columns(4)
                        features = ['valence', 'danceability', 'energy', 'popularity', 'acousticness', 'instrumentalness', 'speechiness', 'tempo']
                        
                        for i, feat in enumerate(features):
                            if feat in selected_song:
                                with char_cols[i % 4]:
                                    value = selected_song[feat]
                                    if feat == 'tempo':
                                        st.metric(feat.capitalize(), f"{value:.0f} BPM")
                                    elif feat == 'popularity':
                                        st.metric(feat.capitalize(), f"{value:.0f}/100")
                                    else:
                                        st.metric(feat.capitalize(), f"{value:.3f}")
                    
                    st.divider()
                    
                    # Obtener recomendaciones
                    with st.spinner("🎵 Buscando canciones similares..."):
                        some_id = selected_song['id']
                        
                        recoms = recommend_by_track_id(
                            df, nn, X_final_norm,
                            some_id,
                            n_recs=n_recs,
                            rerank_by_artist=rerank_by_artist,
                            pop_weight=pop_weight,
                            boost_recency=boost_recency,
                            filter_explicit=filter_explicit,
                            year_range=year_range,
                            min_popularity=min_popularity
                        )
                    
                    if recoms.empty:
                        st.warning("⚠️ No se encontraron recomendaciones con los filtros aplicados. Intenta relajar los criterios.")
                    else:
                        st.subheader(f"🎧 Top {len(recoms)} Recomendaciones")
                        
                        # Formatear el dataframe para mejor visualización
                        display_df = recoms.copy()
                        
                        # Formatear columnas
                        if 'similarity' in display_df.columns:
                            display_df['similarity'] = display_df['similarity'].apply(lambda x: f"{x:.4f}")
                        if 'score' in display_df.columns:
                            display_df['score'] = display_df['score'].apply(lambda x: f"{x:.4f}")
                        if 'popularity' in display_df.columns:
                            display_df['popularity'] = display_df['popularity'].astype(int)
                        
                        # Renombrar columnas para mejor presentación
                        column_names = {
                            'name': 'Canción',
                            'artists': 'Artista',
                            'artist_primary': 'Artista Principal',
                            'year': 'Año',
                            'popularity': 'Popularidad',
                            'explicit_flag': 'Explícito',
                            'similarity': 'Similitud',
                            'score': 'Puntuación'
                        }
                        
                        display_df = display_df.rename(columns=column_names)
                        
                        # Mostrar tabla
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Estadísticas de las recomendaciones
                        st.subheader("📊 Estadísticas de las Recomendaciones")
                        stat_cols = st.columns(4)
                        
                        with stat_cols[0]:
                            avg_sim = recoms['similarity'].mean()
                            st.metric("Similitud promedio", f"{avg_sim:.4f}")
                        
                        with stat_cols[1]:
                            avg_pop = recoms['popularity'].mean()
                            st.metric("Popularidad promedio", f"{avg_pop:.0f}/100")
                        
                        with stat_cols[2]:
                            year_range_result = f"{int(recoms['year'].min())} - {int(recoms['year'].max())}"
                            st.metric("Rango de años", year_range_result)
                        
                        with stat_cols[3]:
                            unique_artists = recoms['artist_primary'].nunique()
                            st.metric("Artistas únicos", unique_artists)
                        
                        # Descargar resultados
                        st.download_button(
                            label="📥 Descargar recomendaciones (CSV)",
                            data=recoms.to_csv(index=False).encode('utf-8'),
                            file_name=f"recomendaciones_{cancion.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"❌ Error al procesar la búsqueda: {str(e)}")
                st.exception(e)
        else:
            st.info("👆 Escribe el nombre de una canción en el campo de búsqueda")

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #b3b3b3; padding: 1rem;'>
        <p>Sistema de recomendación basado en similitud de características de audio usando content-based filtering</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {str(e)}")
    st.info("""
    **Instrucciones:**
    
    1. Asegúrate de tener un archivo `data.csv` en el mismo directorio
    2. El archivo debe contener las columnas necesarias (id, name, artists, year, popularity, etc.)
    3. Ejecuta la aplicación con: `streamlit run app.py`
    """)
    st.exception(e)