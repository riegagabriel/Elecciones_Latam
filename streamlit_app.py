# app.py
# Ejecutar con: streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Candidatos LATAM 2025",
    page_icon="🗳️",
    layout="wide",
)

COLORES_CANDIDATO = {
    "Jeannette Jara Román":   "#E45756",
    "José Antonio Kast":      "#B22222",
    "Franco Parisi":          "#FF7F0E",
    "Johannes Kaiser":        "#8B4513",
    "Evelyn Matthei":         "#F4A9C0",
    'Jorge "Tuto" Quiroga':   "#2CA02C",
    "Rodrigo Paz Pereira":    "#98DF8A",
    "Daniel Noboa":           "#1F77B4",
    "Luisa González":         "#AEC7E8",
}
COLORES_PAIS = {
    "Chile":    "#D62728",
    "Bolivia":  "#2CA02C",
    "Ecuador":  "#1F77B4",
}

NOTA_METODOLOGICA = """
**Nota metodológica · Comparación intra-país**

Los análisis por candidato se realizan dentro de cada país por razones metodológicas:
los contextos electorales son distintos (1ra/2da vuelta, duración del período, sistema político),
los niveles de base de seguidores difieren ampliamente entre países, y el uso de Twitter varía
estructuralmente entre Chile, Bolivia y Ecuador. Comparar candidatos entre países introduciría
sesgos difíciles de controlar. El único análisis multi-país es el gráfico de **Volumen vs Eficiencia**,
que normaliza por tweet y permite una lectura comparada sin esos sesgos.
"""

# ── CARGA DE DATOS ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/tweets_consolidado_20260328_204225.csv")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["fecha"]  = df["created_at"].dt.date
    df["semana"] = (df["created_at"]
                    .dt.to_period("W")
                    .dt.start_time
                    .dt.tz_localize(None))
    df["tipo"] = df.apply(lambda r:
        "Retweet" if r["is_retweet"] else
        "Reply"   if r["is_reply"]   else
        "Quote"   if r["is_quote"]   else
        "Original", axis=1)
    return df

df = load_data()
df_prop = df[~df["is_retweet"]].copy()   # sin retweets para métricas propias

# ── HELPERS ───────────────────────────────────────────────────────────────────
def top_hashtags(series, n=10):
    items = []
    for s in series.dropna():
        items.extend([x.strip().lower() for x in str(s).split("|") if x.strip()])
    return pd.DataFrame(Counter(items).most_common(n), columns=["hashtag", "frecuencia"])

def fmt(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}k"
    return str(n)

def metricas_candidatos(data):
    """Agrega métricas principales por candidato."""
    return (data.groupby(["candidato_nombre", "candidato_pais", "candidato_vuelta"])
            .agg(
                tweets          = ("tweet_id",       "count"),
                likes           = ("like_count",      "sum"),
                rts             = ("retweet_count",   "sum"),
                replies         = ("reply_count",     "sum"),
                views           = ("view_count",      "sum"),
                engagement      = ("engagement_total","sum"),
            )
            .reset_index()
            .assign(
                likes_x_tweet  = lambda d: (d.likes  / d.tweets).round(0).astype(int),
                views_x_tweet  = lambda d: (d.views  / d.tweets).round(0).astype(int),
                eng_x_tweet    = lambda d: (d.engagement / d.tweets).round(0).astype(int),
            ))


# ════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ════════════════════════════════════════════════════════════════════════════
tab_general, tab_chile, tab_bolivia, tab_ecuador = st.tabs([
    "🌎 Vista general",
    "🇨🇱 Chile",
    "🇧🇴 Bolivia",
    "🇪🇨 Ecuador",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 · VISTA GENERAL
# ════════════════════════════════════════════════════════════════════════════
with tab_general:
    st.title("🗳️ Actividad en Twitter · Candidatos Presidenciales LATAM 2025")
    st.markdown(NOTA_METODOLOGICA)
    st.divider()

    # ── KPIs globales ──────────────────────────────────────────────────────
    st.subheader("Totales capturados")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tweets totales",   fmt(len(df)))
    c2.metric("Tweets propios",   fmt(len(df_prop)))
    c3.metric("Likes",            fmt(df_prop["like_count"].sum()))
    c4.metric("Views",            fmt(df_prop["view_count"].sum()))
    c5.metric("Candidatos",       df["candidato_nombre"].nunique())

    st.divider()

    # ── Tabla de contexto electoral ────────────────────────────────────────
    st.subheader("Contexto electoral y datos capturados")

    contexto = (df.groupby(
        ["candidato_pais","candidato_nombre","candidato_vuelta","periodo_busqueda"])
        .agg(
            tweets_capturados = ("tweet_id",      "count"),
            likes_totales     = ("like_count",    "sum"),
            views_totales     = ("view_count",    "sum"),
            seguidores_hoy    = ("author_followers","first"),
        )
        .reset_index()
        .sort_values(["candidato_pais","candidato_nombre"])
        .rename(columns={
            "candidato_pais":     "País",
            "candidato_nombre":   "Candidato",
            "candidato_vuelta":   "Vuelta",
            "periodo_busqueda":   "Período analizado",
            "tweets_capturados":  "Tweets",
            "likes_totales":      "Likes",
            "views_totales":      "Views",
            "seguidores_hoy":     "Seguidores (hoy)",
        })
    )

    # Formatear columnas numéricas para mostrar
    for col in ["Likes","Views","Seguidores (hoy)"]:
        contexto[col] = contexto[col].apply(lambda x: f"{int(x):,}")

    st.dataframe(
        contexto,
        use_container_width=True,
        hide_index=True,
        column_config={
            "País":             st.column_config.TextColumn(width="small"),
            "Vuelta":           st.column_config.TextColumn(width="small"),
            "Tweets":           st.column_config.NumberColumn(width="small"),
        }
    )

    st.divider()

    # ── Tweets capturados por país ─────────────────────────────────────────
    st.subheader("Tweets capturados por país")
    col_a, col_b = st.columns(2)

    tweets_pais = (df_prop.groupby("candidato_pais")
                   .agg(tweets=("tweet_id","count"),
                        likes=("like_count","sum"),
                        views=("view_count","sum"))
                   .reset_index())

    with col_a:
        fig_tp = px.bar(
            tweets_pais,
            x="candidato_pais", y="tweets",
            color="candidato_pais",
            color_discrete_map=COLORES_PAIS,
            text="tweets",
            labels={"candidato_pais": "País", "tweets": "Tweets propios"},
            title="Tweets propios capturados",
        )
        fig_tp.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_tp.update_layout(showlegend=False)
        st.plotly_chart(fig_tp, use_container_width=True)

    with col_b:
        fig_vp = px.bar(
            tweets_pais,
            x="candidato_pais", y="views",
            color="candidato_pais",
            color_discrete_map=COLORES_PAIS,
            text="views",
            labels={"candidato_pais": "País", "views": "Views totales"},
            title="Views totales por país",
        )
        fig_vp.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_vp.update_layout(showlegend=False)
        st.plotly_chart(fig_vp, use_container_width=True)

    st.divider()

    # ── Volumen vs Eficiencia — ÚNICO gráfico multi-país ──────────────────
    st.subheader("Volumen vs Eficiencia (vista multi-país)")
    st.caption(
        "Este es el único análisis que compara candidatos entre países, "
        "porque normaliza el engagement por tweet eliminando el efecto "
        "del volumen de seguidores y del período capturado."
    )

    met_global = metricas_candidatos(df_prop)

    fig_burbuja = px.scatter(
        met_global,
        x="tweets",
        y="likes_x_tweet",
        size="views",
        color="candidato_nombre",
        color_discrete_map=COLORES_CANDIDATO,
        symbol="candidato_pais",
        hover_name="candidato_nombre",
        hover_data={
            "candidato_pais":  True,
            "tweets":          True,
            "likes_x_tweet":   True,
            "views_x_tweet":   True,
            "views":           False,
        },
        title="Volumen (tweets capturados) vs Eficiencia (likes/tweet)",
        labels={
            "tweets":        "N° tweets capturados",
            "likes_x_tweet": "Likes promedio por tweet",
            "candidato_pais":"País",
        },
        size_max=60,
    )
    fig_burbuja.update_layout(legend_title="Candidato", height=480)
    st.plotly_chart(fig_burbuja, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# FUNCIÓN REUTILIZABLE PARA TABS DE PAÍS
# ════════════════════════════════════════════════════════════════════════════
def render_pais(pais: str):
    """Renderiza el análisis completo para un país."""

    data     = df_prop[df_prop["candidato_pais"] == pais].copy()
    data_all = df[df["candidato_pais"] == pais].copy()   # incluye RTs para tipo de tweet
    met      = metricas_candidatos(data).sort_values("likes", ascending=False)
    candidatos = met["candidato_nombre"].tolist()

    # ── KPIs del país ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidatos",    len(candidatos))
    c2.metric("Tweets propios", fmt(len(data)))
    c3.metric("Likes totales",  fmt(data["like_count"].sum()))
    c4.metric("Views totales",  fmt(data["view_count"].sum()))

    st.divider()

    # ── Sección A: Engagement total ────────────────────────────────────────
    st.subheader("Engagement total")
    col1, col2 = st.columns(2)

    with col1:
        fig_likes = px.bar(
            met.sort_values("likes"),
            x="likes", y="candidato_nombre",
            orientation="h",
            color="candidato_nombre",
            color_discrete_map=COLORES_CANDIDATO,
            text="likes",
            title="Likes totales",
            labels={"likes": "Likes", "candidato_nombre": ""},
        )
        fig_likes.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_likes.update_layout(showlegend=False)
        st.plotly_chart(fig_likes, use_container_width=True)

    with col2:
        fig_views = px.bar(
            met.sort_values("views"),
            x="views", y="candidato_nombre",
            orientation="h",
            color="candidato_nombre",
            color_discrete_map=COLORES_CANDIDATO,
            text="views",
            title="Views totales",
            labels={"views": "Views", "candidato_nombre": ""},
        )
        fig_views.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_views.update_layout(showlegend=False)
        st.plotly_chart(fig_views, use_container_width=True)

    st.divider()

    # ── Sección B: Eficiencia ──────────────────────────────────────────────
    st.subheader("Eficiencia por tweet")
    st.caption(
        "Normalizar por tweet permite comparar candidatos con distinto volumen "
        "de publicación dentro del mismo contexto electoral."
    )

    col3, col4 = st.columns(2)
    met_ef = met.sort_values("likes_x_tweet")

    with col3:
        fig_lpt = px.bar(
            met_ef,
            x="likes_x_tweet", y="candidato_nombre",
            orientation="h",
            color="candidato_nombre",
            color_discrete_map=COLORES_CANDIDATO,
            text="likes_x_tweet",
            title="Likes promedio por tweet",
            labels={"likes_x_tweet": "Likes / tweet", "candidato_nombre": ""},
        )
        fig_lpt.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_lpt.update_layout(showlegend=False)
        st.plotly_chart(fig_lpt, use_container_width=True)

    with col4:
        fig_vpt = px.bar(
            met_ef.sort_values("views_x_tweet"),
            x="views_x_tweet", y="candidato_nombre",
            orientation="h",
            color="candidato_nombre",
            color_discrete_map=COLORES_CANDIDATO,
            text="views_x_tweet",
            title="Views promedio por tweet",
            labels={"views_x_tweet": "Views / tweet", "candidato_nombre": ""},
        )
        fig_vpt.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_vpt.update_layout(showlegend=False)
        st.plotly_chart(fig_vpt, use_container_width=True)

    st.divider()

    # ── Sección C: Evolución temporal ──────────────────────────────────────
    st.subheader("Evolución temporal")

    semanal = (data.groupby(["semana","candidato_nombre"])
               .agg(tweets=("tweet_id","count"),
                    likes=("like_count","sum"),
                    views=("view_count","sum"))
               .reset_index())

    metrica_temp = st.radio(
        "Métrica",
        ["likes", "tweets", "views"],
        horizontal=True,
        key=f"radio_temp_{pais}",
    )

    fig_temp = px.line(
        semanal,
        x="semana", y=metrica_temp,
        color="candidato_nombre",
        color_discrete_map=COLORES_CANDIDATO,
        markers=True,
        title=f"{metrica_temp.capitalize()} por semana — {pais}",
        labels={
            "semana":           "Semana",
            metrica_temp:       metrica_temp.capitalize(),
            "candidato_nombre": "Candidato",
        },
    )
    fig_temp.update_layout(legend_title="Candidato", height=380)
    st.plotly_chart(fig_temp, use_container_width=True)

    st.divider()

    # ── Sección D: Tipo de tweet ───────────────────────────────────────────
    st.subheader("Composición de tweets")

    tipos_cand = (data_all.groupby(["candidato_nombre","tipo"])
                  .size().reset_index(name="n"))

    col5, col6 = st.columns(2)

    with col5:
        fig_tipo = px.bar(
            tipos_cand,
            x="candidato_nombre", y="n",
            color="tipo", barmode="stack",
            color_discrete_map={
                "Original": "#1F77B4",
                "Reply":    "#FF7F0E",
                "Retweet":  "#2CA02C",
                "Quote":    "#9467BD",
            },
            title="Tipo de tweet por candidato",
            labels={"candidato_nombre":"","n":"N° tweets","tipo":"Tipo"},
        )
        fig_tipo.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig_tipo, use_container_width=True)

    with col6:
        # Pie de proporciones globales del país
        tipos_pais = data_all["tipo"].value_counts().reset_index()
        tipos_pais.columns = ["tipo","n"]
        fig_pie = px.pie(
            tipos_pais, values="n", names="tipo",
            title=f"Proporción global — {pais}",
            color_discrete_map={
                "Original": "#1F77B4",
                "Reply":    "#FF7F0E",
                "Retweet":  "#2CA02C",
                "Quote":    "#9467BD",
            },
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Sección E: Hashtags ────────────────────────────────────────────────
    st.subheader("Hashtags más usados")

    col7, col8 = st.columns([1, 2])

    with col7:
        candidato_sel = st.selectbox(
            "Ver hashtags de:",
            ["Todos"] + candidatos,
            key=f"ht_sel_{pais}",
        )

    with col8:
        n_tags = st.slider("N° de hashtags", 5, 20, 10, key=f"ht_n_{pais}")

    ht_data = (data if candidato_sel == "Todos"
               else data[data["candidato_nombre"] == candidato_sel])
    ht_df = top_hashtags(ht_data["hashtags"], n=n_tags)

    if not ht_df.empty:
        color_ht = (COLORES_PAIS[pais] if candidato_sel == "Todos"
                    else COLORES_CANDIDATO.get(candidato_sel, "#1F77B4"))
        fig_ht = px.bar(
            ht_df.sort_values("frecuencia"),
            x="frecuencia", y="hashtag",
            orientation="h",
            text="frecuencia",
            title=f"Top {n_tags} hashtags — {candidato_sel}",
            labels={"frecuencia":"Frecuencia","hashtag":""},
            color_discrete_sequence=[color_ht],
        )
        fig_ht.update_traces(textposition="outside")
        fig_ht.update_layout(height=max(300, n_tags * 30))
        st.plotly_chart(fig_ht, use_container_width=True)
    else:
        st.info("Sin hashtags para esta selección.")

    st.divider()

    # ── Sección F: Top tweets ──────────────────────────────────────────────
    st.subheader("Tweets más virales")

    col9, col10 = st.columns([1, 2])
    with col9:
        cand_top = st.selectbox(
            "Filtrar por candidato:",
            ["Todos"] + candidatos,
            key=f"top_sel_{pais}",
        )
    with col10:
        metrica_top = st.radio(
            "Ordenar por:",
            ["like_count","view_count","retweet_count"],
            format_func=lambda x: {"like_count":"Likes",
                                   "view_count":"Views",
                                   "retweet_count":"Retweets"}[x],
            horizontal=True,
            key=f"top_met_{pais}",
        )

    top_data = (data if cand_top == "Todos"
                else data[data["candidato_nombre"] == cand_top])

    top5 = (top_data.sort_values(metrica_top, ascending=False)
            .head(5)[["candidato_nombre","created_at","like_count",
                      "retweet_count","view_count","text","tweet_url"]]
            .reset_index(drop=True))

    for _, row in top5.iterrows():
        with st.container(border=True):
            ca, cb, cc, cd = st.columns([2,1,1,1])
            ca.markdown(f"**{row['candidato_nombre']}**  \n"
                        f"🗓️ {str(row['created_at'])[:10]}")
            cb.metric("Likes",    fmt(row["like_count"]))
            cc.metric("RTs",      fmt(row["retweet_count"]))
            cd.metric("Views",    fmt(row["view_count"]))
            st.markdown(f"> {str(row['text'])[:200]}{'…' if len(str(row['text']))>200 else ''}")
            st.markdown(f"[Ver tweet ↗]({row['tweet_url']})")


# ════════════════════════════════════════════════════════════════════════════
# RENDERIZAR TABS DE PAÍS
# ════════════════════════════════════════════════════════════════════════════
with tab_chile:
    st.title("🇨🇱 Chile — Elecciones presidenciales 2025")
    st.caption("Período: 17 sept – 14 dic 2025 · 1ra y 2da vuelta")
    render_pais("Chile")

with tab_bolivia:
    st.title("🇧🇴 Bolivia — Elecciones presidenciales 2025")
    st.caption("Período: 13 jul – 19 oct 2025 · 2da vuelta")
    render_pais("Bolivia")

with tab_ecuador:
    st.title("🇪🇨 Ecuador — Elecciones presidenciales 2025")
    st.caption("Período: 5 ene – 13 abr 2025 · 2da vuelta")
    render_pais("Ecuador")
