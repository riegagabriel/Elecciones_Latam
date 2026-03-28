# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

# CONFIG
st.set_page_config(page_title="Observatorio LATAM", layout="wide")

# COLORES
COLORES_CANDIDATO = {
    "Jeannette Jara Román": "#E45756",
    "José Antonio Kast": "#B22222",
    "Franco Parisi": "#FF7F0E",
    "Johannes Kaiser": "#8B4513",
    "Evelyn Matthei": "#F4A9C0",
    'Jorge "Tuto" Quiroga': "#2CA02C",
    "Rodrigo Paz Pereira": "#98DF8A",
    "Daniel Noboa": "#1F77B4",
    "Luisa González": "#AEC7E8",
}

# DATA
@st.cache_data
def load_data():
    df = pd.read_csv("data/tweets_consolidado_20260328_204225.csv")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["semana"] = df["created_at"].dt.to_period("W").dt.start_time
    return df

df = load_data()
df_prop = df[~df["is_retweet"]]

# HELPERS
def fmt(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n/1_000:.0f}k"
    return str(int(n))

def metricas(data):
    return (data.groupby(["candidato_nombre","candidato_pais"])
            .agg(tweets=("tweet_id","count"),
                 likes=("like_count","sum"),
                 views=("view_count","sum"))
            .reset_index()
            .assign(
                likes_x_tweet=lambda d: d.likes/d.tweets,
                views_x_tweet=lambda d: d.views/d.tweets
            ))

# TABS
tab1, tab2, tab3 = st.tabs(["🧭 Resumen","🌎 Países","🧠 Candidato"])

# =========================
# TAB 1: RESUMEN
# =========================
with tab1:
    st.title("🗳️ Observatorio de Comunicación Política")

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tweets", fmt(len(df_prop)))
    c2.metric("Likes", fmt(df_prop["like_count"].sum()))
    c3.metric("Views", fmt(df_prop["view_count"].sum()))
    c4.metric("Candidatos", df["candidato_nombre"].nunique())

    st.divider()

    # INSIGHTS AUTOMÁTICOS
    met = metricas(df_prop)

    top_ef = met.sort_values("likes_x_tweet", ascending=False).iloc[0]
    top_vol = met.sort_values("tweets", ascending=False).iloc[0]

    st.subheader("📌 Claves del periodo")
    st.markdown(f"""
    - **Mayor eficiencia:** {top_ef['candidato_nombre']}
    - **Mayor volumen:** {top_vol['candidato_nombre']}
    - No siempre quien más publica genera mayor impacto.
    """)

    # BURBUJA
    fig = px.scatter(
        met,
        x="tweets",
        y="likes_x_tweet",
        size="views",
        color="candidato_nombre",
        color_discrete_map=COLORES_CANDIDATO,
        title="Volumen vs Eficiencia"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2: PAÍSES
# =========================
with tab2:
    st.title("🌎 Comparación por país")

    pais = st.selectbox("Selecciona país", df["candidato_pais"].unique())
    data = df_prop[df_prop["candidato_pais"] == pais]
    met = metricas(data)

    # INSIGHT
    top = met.sort_values("likes_x_tweet", ascending=False).iloc[0]

    st.markdown(f"""
    ### 📌 Lectura clave

    En **{pais}**, **{top['candidato_nombre']}** lidera en eficiencia comunicacional.
    """)

    # GRAFICO 1: EFICIENCIA
    fig1 = px.bar(
        met.sort_values("likes_x_tweet"),
        x="likes_x_tweet",
        y="candidato_nombre",
        orientation="h",
        color="candidato_nombre",
        color_discrete_map=COLORES_CANDIDATO,
        title="Likes por tweet"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # GRAFICO 2: TEMPORAL
    semanal = (data.groupby(["semana","candidato_nombre"])
               .agg(likes=("like_count","sum"))
               .reset_index())

    fig2 = px.line(
        semanal,
        x="semana",
        y="likes",
        color="candidato_nombre",
        title="Evolución semanal"
    )
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 3: CANDIDATO
# =========================
with tab3:
    st.title("🧠 Perfil de candidato")

    cand = st.selectbox("Selecciona candidato", df["candidato_nombre"].unique())
    data = df_prop[df_prop["candidato_nombre"] == cand]

    tweets = len(data)
    likes = data["like_count"].sum()
    lpt = likes / tweets if tweets > 0 else 0

    st.markdown(f"""
    ### 📌 Perfil comunicacional

    **{cand}** ha publicado **{tweets} tweets**  
    Promedio de **{int(lpt)} likes por tweet**

    Estrategia:
    - {"Alto volumen" if tweets > df_prop.shape[0]/5 else "Bajo volumen"}
    - {"Alto engagement" if lpt > df_prop["like_count"].mean() else "Engagement moderado"}
    """)

    # TEMPORAL
    semanal = (data.groupby("semana")
               .agg(likes=("like_count","sum"))
               .reset_index())

    fig1 = px.line(semanal, x="semana", y="likes", title="Evolución")
    st.plotly_chart(fig1, use_container_width=True)

    # TIPO TWEET
    tipos = data["is_reply"].value_counts().reset_index()
    fig2 = px.pie(tipos, values="count", names="index", title="Tipo de interacción")
    st.plotly_chart(fig2, use_container_width=True)

    # TOP TWEETS
    st.subheader("🔥 Tweets más virales")

    top = data.sort_values("like_count", ascending=False).head(3)

    for _, row in top.iterrows():
        st.markdown(f"""
        **{row['created_at'].date()}**  
        ❤️ {fmt(row['like_count'])} | 🔁 {fmt(row['retweet_count'])}

        > {row['text'][:200]}...
        """)
