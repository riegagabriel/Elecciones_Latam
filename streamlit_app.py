# app.py
# Ejecutar con: streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import glob

# ── CONFIG ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Observatorio LATAM 2025",
    page_icon="🗳️",
    layout="wide"
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
    "Chile":   "#D62728",
    "Bolivia": "#2CA02C",
    "Ecuador": "#1F77B4",
}

# ── HITOS ELECTORALES para anotaciones en gráfico ─────────────────────────────
HITOS = {
    "Ecuador": [
        {"fecha": "2025-02-09", "label": "1ª vuelta"},
        {"fecha": "2025-03-23", "label": "Debate 2ª vuelta"},
        {"fecha": "2025-04-13", "label": "2ª vuelta"},
    ],
    "Bolivia": [
        {"fecha": "2025-08-17", "label": "1ª vuelta"},
        {"fecha": "2025-10-12", "label": "Debate balotaje"},
        {"fecha": "2025-10-19", "label": "2ª vuelta"},
    ],
    "Chile": [
        {"fecha": "2025-09-10", "label": "Debate CHV"},
        {"fecha": "2025-11-04", "label": "Debate ARCHI"},
        {"fecha": "2025-11-10", "label": "Debate ANATEL"},
        {"fecha": "2025-11-16", "label": "1ª vuelta"},
        {"fecha": "2025-12-14", "label": "2ª vuelta"},
    ],
}

CONTEXTO = {
    "Ecuador": {
        "resultado": "Noboa se impuso con un 55,63% en la segunda vuelta (13 abr 2025).",
        "narrativa": (
            "El calendario del CNE estableció el inicio oficial de campaña el **5 de enero de 2025**. "
            "La primera vuelta se celebró el **9 de febrero** y la segunda vuelta el **13 de abril**. "
            "El debate presidencial del **23 de marzo** fue el hito central: González interpeló a Noboa "
            "por incumplimiento de promesas, mientras Noboa posicionó la narrativa del *'Nuevo Ecuador'* "
            "frente al *'pasado fallido'*. En X, Noboa priorizó hitos de gestión en seguridad y evitó "
            "la confrontación directa; González utilizó la plataforma para denunciar la falta de un plan "
            "de seguridad en un contexto de alta violencia."
        ),
        "debate_clave": "23 de marzo de 2025 — Debate presidencial 2ª vuelta",
        "periodo": "5 ene – 13 abr 2025",
        "vuelta": "2da vuelta",
    },
    "Bolivia": {
        "resultado": "Paz Pereira resultó electo (19 oct 2025), marcando el mayor giro hacia la centroderecha en dos décadas.",
        "narrativa": (
            "Bolivia vivió el desplome electoral más significativo del MAS en dos décadas. "
            "La primera vuelta (**17 de agosto**) fragmentó el voto oficialista, permitiendo el pase "
            "de Paz Pereira y Quiroga al balotaje del **19 de octubre**. "
            "El debate del **12 de octubre** fue el punto de inflexión: ambos se comprometieron a "
            "respetar los resultados. Quiroga utilizó X como plataforma de denuncia internacional; "
            "Paz se posicionó como candidato de la *'renovación'*. Su moderación en X frente al tono "
            "más combativo de Quiroga le habría permitido capturar el voto útil."
        ),
        "debate_clave": "12 de octubre de 2025 — Debate presidencial 2ª vuelta",
        "periodo": "13 jul – 19 oct 2025",
        "vuelta": "2da vuelta",
    },
    "Chile": {
        "resultado": "Kast se impuso con un 58,17% en el balotaje del 14 de diciembre de 2025.",
        "narrativa": (
            "Las elecciones de Chile representaron el péndulo político más extremo de su historia reciente. "
            "La primera vuelta (**16 de noviembre**) dejó fuera a Matthei y Parisi. El balotaje del "
            "**14 de diciembre** enfrentó a la oficialista Jara contra Kast. "
            "La campaña tuvo tres debates clave: **10 de septiembre** en Chilevisión, "
            "**4 de noviembre** en ARCHI y **10 de noviembre** en ANATEL. "
            "En X, Kast lideró el engagement con narrativas de orden y seguridad; Matthei buscó "
            "gobernabilidad institucional; Jara defendió las reformas sociales pero fue interpelada "
            "constantemente por la continuidad del programa oficialista."
        ),
        "debate_clave": "10 nov 2025 — Debate ANATEL (último antes de 1ª vuelta)",
        "periodo": "17 sept – 14 dic 2025",
        "vuelta": "1ra y 2da vuelta",
    },
}

# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/tweets_consolidado_20260328_204225.csv")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["semana"] = (
        df["created_at"]
        .dt.to_period("W")
        .dt.start_time
        .dt.tz_localize(None)
    )
    df["tipo"] = df.apply(lambda r:
        "Retweet"  if r["is_retweet"] else
        "Reply"    if r["is_reply"]   else
        "Quote"    if r["is_quote"]   else
        "Original", axis=1)
    return df

@st.cache_data
def load_opinion():
    try:
        df_op = pd.read_csv("data/opinion_consolidado_20260329_172545.csv")
        df_op["created_at"] = pd.to_datetime(df_op["created_at"], utc=True, errors="coerce")
        df_op["semana"] = (
            df_op["created_at"]
            .dt.to_period("W")
            .dt.start_time
            .dt.tz_localize(None)
        )
        return df_op
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_perfiles():
    archivos = glob.glob("data/perfiles_*.csv")
    if not archivos:
        return pd.DataFrame()
    return pd.read_csv(sorted(archivos)[-1])

df       = load_data()
df_prop  = df[~df["is_retweet"]].copy()
df_op    = load_opinion()
df_perf  = load_perfiles()
HAY_OPINION  = not df_op.empty
HAY_PERFILES = not df_perf.empty

# ── HELPERS DE PERFILES ────────────────────────────────────────────────────────
# Detectar columnas disponibles una sola vez al inicio
_foto_col = None
_id_col   = None
_seg_col  = None

if HAY_PERFILES:
    for c in ["foto_perfil_url", "profilePicture", "profile_image_url"]:
        if c in df_perf.columns:
            _foto_col = c
            break
    for c in ["nombre_real", "nombre_display", "username"]:
        if c in df_perf.columns:
            _id_col = c
            break
    for c in ["seguidores", "followers", "seguidores_snapshot_hoy"]:
        if c in df_perf.columns:
            _seg_col = c
            break

def _buscar_perfil(nombre):
    """Busca la fila del candidato en df_perf probando múltiples columnas de id."""
    if not HAY_PERFILES:
        return None
    # Intentar match exacto en todas las columnas candidatas
    for col in ["nombre_real", "nombre_display", "username", "name"]:
        if col in df_perf.columns:
            fila = df_perf[df_perf[col] == nombre]
            if not fila.empty:
                return fila.iloc[0]
    # Fallback: match parcial en nombre_real o nombre_display
    for col in ["nombre_real", "nombre_display"]:
        if col in df_perf.columns:
            fila = df_perf[df_perf[col].str.contains(
                nombre.split()[0], case=False, na=False
            )]
            if not fila.empty:
                return fila.iloc[0]
    return None

def get_foto(nombre):
    if not HAY_PERFILES or _foto_col is None:
        return None
    perfil = _buscar_perfil(nombre)
    if perfil is None:
        return None
    url = perfil[_foto_col]
    if not pd.notna(url) or not str(url).startswith("http"):
        return None
    return str(url).replace("_normal.", "_400x400.")

def get_seguidores(nombre):
    if not HAY_PERFILES or _seg_col is None:
        return None
    perfil = _buscar_perfil(nombre)
    if perfil is None:
        return None
    val = perfil[_seg_col]
    return int(val) if pd.notna(val) else None

# ── HELPERS GENERALES ──────────────────────────────────────────────────────────
def fmt(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}k"
    return str(int(n))

def top_hashtags(series, n=10):
    tags = []
    for s in series.dropna():
        tags += [x.strip().lower() for x in str(s).split("|") if x.strip()]
    return pd.DataFrame(Counter(tags).most_common(n), columns=["hashtag", "freq"])

def metricas(data):
    return (
        data.groupby("candidato_nombre")
        .agg(
            tweets = ("tweet_id",   "count"),
            likes  = ("like_count", "sum"),
            views  = ("view_count", "sum"),
        )
        .reset_index()
        .assign(likes_x_tweet=lambda d: (d.likes / d.tweets).round(0).astype(int))
    )

def nota(texto):
    st.markdown(
        f"<div style='background:#f0f2f6;border-left:3px solid #4a90d9;"
        f"padding:8px 12px;border-radius:4px;font-size:0.87em;color:#444;"
        f"margin-bottom:10px'>💡 {texto}</div>",
        unsafe_allow_html=True,
    )

def render_contexto_pais(pais):
    ctx = CONTEXTO.get(pais, {})
    if not ctx:
        return
    color = COLORES_PAIS[pais]
    st.markdown(
        f"<div style='background:{color}12;border-left:4px solid {color};"
        f"padding:12px 16px;border-radius:6px;margin-bottom:12px'>"
        f"<p style='margin:0 0 4px;font-size:0.78em;color:{color};font-weight:600;letter-spacing:.05em'>"
        f"🏆 RESULTADO · {ctx['periodo']} · {ctx['vuelta']}</p>"
        f"<p style='margin:0 0 6px;font-weight:600;font-size:0.97em'>{ctx['resultado']}</p>"
        f"<p style='margin:0;font-size:0.83em;color:#555'>"
        f"📅 Debate clave: {ctx['debate_clave']}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    with st.expander("📖 Contexto electoral completo"):
        st.markdown(ctx["narrativa"])

# ── WORDCLOUD ──────────────────────────────────────────────────────────────────
STOPWORDS_ES = {
    "de","la","el","en","y","a","los","del","se","las","por","un","para",
    "con","una","su","al","lo","como","más","pero","sus","le","ya","o",
    "este","sí","porque","esta","entre","cuando","muy","sin","sobre",
    "también","me","hasta","hay","donde","han","yo","él","ella","nos",
    "todo","estos","estas","fue","son","ser","tiene","tenemos","que","es",
    "no","si","te","mi","tu","http","https","t","co","amp","rt","via",
    "hoy","ayer","así","bien","gran","cada","hacer","puede","nuestro",
    "nuestra","nuestros","nuestras","está","están","tener","solo","todos",
    "todas","otro","otra","años","Chile","Bolivia","Ecuador","país",
    "gobierno","presidente","presidenta","hemos","sido","esto",
}

@st.cache_data
def generar_wordcloud(candidato: str, color: str) -> BytesIO:
    textos = df_prop[df_prop["candidato_nombre"] == candidato]["text"].dropna()
    def limpiar(t):
        t = re.sub(r"http\S+", "", t)
        t = re.sub(r"@\w+", "", t)
        t = re.sub(r"#(\w+)", r"\1", t)
        t = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", " ", t)
        return t.lower()
    corpus = " ".join(limpiar(t) for t in textos)
    wc = WordCloud(
        width=700, height=320, background_color="white",
        color_func=lambda *a, **k: color,
        stopwords=STOPWORDS_ES, min_word_length=4,
        max_words=60, collocations=False, prefer_horizontal=0.85,
    )
    wc.generate(corpus)
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ── GRÁFICO TEMPORAL CON HITOS ─────────────────────────────────────────────────
def render_timeline_hitos(pais: str, data: pd.DataFrame, ctx: str = "tab"):
    """Línea de likes semanales por candidato con líneas verticales en hitos electorales."""
    semanal = (
        data.groupby(["semana", "candidato_nombre"])
        .agg(likes=("like_count","sum"), tweets=("tweet_id","count"))
        .reset_index()
    )

    metrica = st.radio(
        "Ver por:",
        ["likes", "tweets"],
        format_func=lambda x: {"likes": "Likes", "tweets": "Volumen de tweets"}[x],
        horizontal=True,
        key=f"timeline_metric_{pais}_{ctx}",
    )

    fig = px.line(
        semanal, x="semana", y=metrica,
        color="candidato_nombre",
        color_discrete_map=COLORES_CANDIDATO,
        markers=True,
        labels={
            "semana": "",
            metrica: "Likes semanales" if metrica == "likes" else "Tweets semanales",
            "candidato_nombre": "",
        },
    )
    fig.update_layout(
        height=400,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=20, b=60),
        hovermode="x unified",
    )

    # Añadir líneas verticales en hitos electorales
    hitos_pais = HITOS.get(pais, [])
    for h in hitos_pais:
        fig.add_vline(
            x=h["fecha"],
            line_dash="dash",
            line_color="rgba(0,0,0,0.25)",
            line_width=1.5,
        )
        fig.add_annotation(
            x=h["fecha"],
            y=1,
            yref="paper",
            text=h["label"],
            showarrow=False,
            font=dict(size=10, color="#555"),
            textangle=-55,
            xanchor="left",
            yanchor="top",
        )

    st.plotly_chart(fig, use_container_width=True, key=f"timeline_{pais}_{ctx}")
    nota(
        "Las líneas verticales marcan hitos electorales clave: debates y fechas de votación. "
        "Observa si los picos de engagement coinciden con estos momentos o si la conversación "
        "se anticipa o retrasa respecto a los eventos."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN: ¿QUÉ DICE LA GENTE?
# ══════════════════════════════════════════════════════════════════════════════
def render_opinion(pais: str):
    if not HAY_OPINION:
        return
    col_nombre = "candidato_nombre" if "candidato_nombre" in df_op.columns else "candidato"
    data_op = df_op[df_op["candidato_pais"] == pais].copy()
    if data_op.empty:
        return

    st.divider()
    st.subheader("🗣️ ¿Y qué dice la gente?")
    st.caption(
        "Muestra de tweets de ciudadanos que mencionaron o hablaron de los candidatos "
        "durante el período. No es exhaustiva — sirve para identificar temas y conversaciones, "
        "no para estimar volúmenes totales de menciones."
    )

    candidatos_op = sorted(data_op[col_nombre].unique().tolist())
    candidato_sel = st.selectbox("Ver conversación sobre:", candidatos_op, key=f"op_{pais}")
    d     = data_op[data_op[col_nombre] == candidato_sel]
    color = COLORES_CANDIDATO.get(candidato_sel, COLORES_PAIS[pais])

    col_autor = "autor_username" if "autor_username" in d.columns else "author_username"
    c1, c2, c3 = st.columns(3)
    c1.metric("Tweets en la muestra",  fmt(len(d)))
    c2.metric("Personas distintas",
              fmt(d[col_autor].nunique()) if col_autor in d.columns else "—")
    c3.metric("Likes acumulados", fmt(d["like_count"].sum()))
    nota("Estos números reflejan la muestra capturada, no el universo total de menciones en X.")

    st.markdown("**¿Cuándo habló más la gente de este candidato?**")
    semanal_op = d.groupby("semana").agg(tweets=("tweet_id","count")).reset_index()
    fig_ev = px.area(
        semanal_op, x="semana", y="tweets",
        labels={"semana": "Semana", "tweets": "Tweets capturados"},
        color_discrete_sequence=[color],
    )
    fig_ev.update_layout(showlegend=False, height=220, margin=dict(t=10, b=10))
    st.plotly_chart(fig_ev, use_container_width=True, key=f"op_ev_{pais}_{candidato_sel}")
    nota("Los picos indican semanas con mayor conversación ciudadana. Pueden coincidir con debates o hitos electorales.")

    if "hashtags" in d.columns:
        st.markdown("**¿Con qué hashtags habla la gente de este candidato?**")
        ht_op = top_hashtags(d["hashtags"], n=8)
        if not ht_op.empty:
            fig_ht = px.bar(
                ht_op.sort_values("freq"),
                x="freq", y="hashtag", orientation="h", text="freq",
                labels={"freq": "Usos", "hashtag": ""},
                color_discrete_sequence=[color],
            )
            fig_ht.update_traces(textposition="outside")
            fig_ht.update_layout(showlegend=False, height=280, margin=dict(t=10))
            st.plotly_chart(fig_ht, use_container_width=True, key=f"op_ht_{pais}_{candidato_sel}")

    st.markdown("**Tweets ciudadanos con más likes sobre este candidato**")
    col_rt = "retweet_count" if "retweet_count" in d.columns else None
    top_op = d.sort_values("like_count", ascending=False).head(3).reset_index(drop=True)
    for _, row in top_op.iterrows():
        with st.container(border=True):
            autor = row.get("autor_username", row.get("author_username", "—"))
            ca, cb, cc = st.columns([4, 1, 1])
            ca.markdown(f"**@{autor}** · {str(row['created_at'])[:10]}")
            cb.metric("❤️", fmt(row["like_count"]))
            cc.metric("🔁", fmt(row[col_rt]) if col_rt else "—")
            st.markdown(f"> {str(row['text'])[:200]}{'…' if len(str(row['text'])) > 200 else ''}")

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN REUTILIZABLE POR PAÍS
# ══════════════════════════════════════════════════════════════════════════════
def render_pais(pais: str):
    data = df_prop[df_prop["candidato_pais"] == pais].copy()
    met  = metricas(data)
    candidatos = met["candidato_nombre"].tolist()

    c1, c2, c3 = st.columns(3)
    c1.metric("Tweets propios", fmt(len(data)))
    c2.metric("Likes totales",  fmt(data["like_count"].sum()))
    c3.metric("Views totales",  fmt(data["view_count"].sum()))

    st.divider()

    st.subheader("Engagement y eficiencia")
    nota("Izquierda: impacto bruto acumulado. Derecha: rendimiento promedio por tweet. Cuando el ranking difiere, el candidato más activo no es el que mejor conecta.")
    col1, col2 = st.columns(2)
    with col1:
        fig_likes = px.bar(
            met.sort_values("likes"),
            x="likes", y="candidato_nombre", orientation="h",
            color="candidato_nombre", color_discrete_map=COLORES_CANDIDATO,
            text="likes", title="Likes totales",
            labels={"likes":"Likes","candidato_nombre":""},
        )
        fig_likes.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_likes.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_likes, use_container_width=True, key=f"likes_{pais}")
    with col2:
        fig_lpt = px.bar(
            met.sort_values("likes_x_tweet"),
            x="likes_x_tweet", y="candidato_nombre", orientation="h",
            color="candidato_nombre", color_discrete_map=COLORES_CANDIDATO,
            text="likes_x_tweet", title="Likes por tweet",
            labels={"likes_x_tweet":"Likes/tweet","candidato_nombre":""},
        )
        fig_lpt.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_lpt.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_lpt, use_container_width=True, key=f"lpt_{pais}")

    st.divider()

    st.subheader("Evolución semanal y hitos electorales")
    nota("Las líneas verticales marcan debates y fechas de votación. ¿Coinciden los picos con los eventos clave?")
    render_timeline_hitos(pais, data, ctx="pais")

    st.divider()

    st.subheader("Hashtags más usados")
    nota("Revelan los ejes temáticos de campaña y los eventos en que participó activamente.")
    candidato_ht = st.selectbox("Ver hashtags de:", candidatos, key=f"ht_{pais}")
    ht = top_hashtags(data[data["candidato_nombre"] == candidato_ht]["hashtags"], n=10)
    if not ht.empty:
        color_ht = COLORES_CANDIDATO.get(candidato_ht, COLORES_PAIS[pais])
        fig_ht = px.bar(
            ht.sort_values("freq"),
            x="freq", y="hashtag", orientation="h", text="freq",
            labels={"freq":"Usos","hashtag":""},
            color_discrete_sequence=[color_ht],
        )
        fig_ht.update_traces(textposition="outside")
        fig_ht.update_layout(showlegend=False, height=max(280, len(ht)*32))
        st.plotly_chart(fig_ht, use_container_width=True, key=f"ht_{pais}_{candidato_ht}")

    st.divider()

    st.subheader("Tweets más virales")
    nota("El contenido con mayor respuesta — en las propias palabras del candidato.")
    candidato_top = st.selectbox("Ver tweets de:", ["Todos"] + candidatos, key=f"top_{pais}")
    top_data = data if candidato_top == "Todos" else data[data["candidato_nombre"] == candidato_top]
    top5 = top_data.sort_values("like_count", ascending=False).head(5).reset_index(drop=True)
    for _, row in top5.iterrows():
        with st.container(border=True):
            ca, cb, cc, cd = st.columns([3,1,1,1])
            ca.markdown(f"**{row['candidato_nombre']}** · {str(row['created_at'])[:10]}")
            cb.metric("❤️", fmt(row["like_count"]))
            cc.metric("🔁", fmt(row["retweet_count"]))
            cd.metric("👁️", fmt(row["view_count"]))
            st.markdown(f"> {str(row['text'])[:220]}{'…' if len(str(row['text']))>220 else ''}")
            if pd.notna(row.get("tweet_url","")):
                st.markdown(f"[Ver en X ↗]({row['tweet_url']})")

    render_opinion(pais)


# ══════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════
st.title("🗳️ Observatorio LATAM 2025")

tab_general, tab_chile, tab_bolivia, tab_ecuador = st.tabs([
    "🌎 General", "🇨🇱 Chile", "🇧🇴 Bolivia", "🇪🇨 Ecuador",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB GENERAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_general:
    st.subheader("Resumen general")
    st.markdown(
        "Actividad en Twitter/X de los principales candidatos presidenciales de "
        "Chile, Bolivia y Ecuador durante sus respectivos períodos electorales de 2025."
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Países",         3)
    c2.metric("Candidatos",     df["candidato_nombre"].nunique())
    c3.metric("Tweets propios", fmt(len(df_prop)))
    c4.metric("Likes totales",  fmt(df_prop["like_count"].sum()))
    c5.metric("Views totales",  fmt(df_prop["view_count"].sum()))
    c6.metric("Retweets",       fmt(df_prop["retweet_count"].sum()))

    st.divider()

    # ── Tarjetas de candidatos ─────────────────────────────────────────────
    st.subheader("Los candidatos")

    met_cands = metricas(df_prop).merge(
        df_prop[["candidato_nombre","candidato_pais","candidato_vuelta","periodo_busqueda"]]
        .drop_duplicates("candidato_nombre"),
        on="candidato_nombre", how="left"
    )

    # Selector de país con pills (fallback a radio si versión antigua)
    opciones_pais = ["🇨🇱  Chile", "🇧🇴  Bolivia", "🇪🇨  Ecuador"]
    mapa_pais     = {"🇨🇱  Chile":"Chile", "🇧🇴  Bolivia":"Bolivia", "🇪🇨  Ecuador":"Ecuador"}
    try:
        sel = st.pills("Selecciona un país:", opciones_pais, default="🇨🇱  Chile", key="pills_pais")
        pais_sel = mapa_pais.get(sel, "Chile")
    except AttributeError:
        sel = st.radio("Selecciona un país:", opciones_pais, horizontal=True, key="radio_pais")
        pais_sel = mapa_pais.get(sel, "Chile")

    color_pais = COLORES_PAIS[pais_sel]
    render_contexto_pais(pais_sel)

    # Tarjetas
    cands_pais = met_cands[met_cands["candidato_pais"] == pais_sel].sort_values("likes", ascending=False)
    cols_cands = st.columns(len(cands_pais))
    for col, (_, row) in zip(cols_cands, cands_pais.iterrows()):
        nombre     = row["candidato_nombre"]
        foto       = get_foto(nombre)
        seguidores = get_seguidores(nombre)
        with col:
            with st.container(border=True):
                # Usar st.image con URL directa — única forma que funciona en Streamlit
                if foto:
                    st.image(foto, width=80)
                st.markdown(f"**{nombre}**")
                st.caption(row["candidato_vuelta"])
                st.divider()
                m1, m2 = st.columns(2)
                m1.metric("❤️ Likes",  fmt(row["likes"]))
                m2.metric("📄 Tweets", fmt(row["tweets"]))
                m1.metric("⚡/tweet",  fmt(row["likes_x_tweet"]))
                m2.metric("👁️ Views",  fmt(row["views"]))
                if seguidores:
                    st.caption(f"👥 {fmt(seguidores)} seguidores")

    st.divider()

    # ── Timeline con hitos — filtrado por país seleccionado ────────────────
    st.subheader("Evolución semanal y hitos electorales")
    nota(
        "Likes semanales de cada candidato. Las líneas verticales marcan debates y votaciones. "
        "Cambia de país arriba para ver el contexto de cada elección."
    )
    data_pais_sel = df_prop[df_prop["candidato_pais"] == pais_sel].copy()
    render_timeline_hitos(pais_sel, data_pais_sel, ctx="home")

    st.divider()

    # ── Volumen vs eficiencia ──────────────────────────────────────────────
    st.subheader("Volumen vs eficiencia")
    nota(
        "Única vista multi-país. Normaliza por likes/tweet para comparar candidatos "
        "con contextos electorales distintos. "
        "Eje X = cuánto publica · Eje Y = cuánto resuena · Tamaño = views totales."
    )
    fig_burbuja = px.scatter(
        met_cands,
        x="tweets", y="likes_x_tweet", size="views",
        color="candidato_pais", color_discrete_map=COLORES_PAIS,
        symbol="candidato_pais", hover_name="candidato_nombre",
        hover_data={"candidato_pais":True,"tweets":True,"likes_x_tweet":True,"views":False},
        labels={"tweets":"N° tweets","likes_x_tweet":"Likes promedio por tweet","candidato_pais":"País"},
        size_max=55,
    )
    fig_burbuja.update_layout(height=420, legend_title="País")
    st.plotly_chart(fig_burbuja, use_container_width=True, key="burbuja_global")

    st.divider()

    # ── Tabla de contexto ──────────────────────────────────────────────────
    st.subheader("Tabla de contexto")
    nota("Período analizado, vuelta electoral y métricas clave de cada candidato.")
    tabla = met_cands[["candidato_pais","candidato_nombre","candidato_vuelta",
                        "periodo_busqueda","tweets","likes","views","likes_x_tweet"]].copy()
    tabla = tabla.sort_values(["candidato_pais","likes"], ascending=[True,False])
    tabla.columns = ["País","Candidato","Vuelta","Período","Tweets","Likes","Views","Likes/tweet"]
    for c in ["Likes","Views"]:
        tabla[c] = tabla[c].apply(lambda x: f"{int(x):,}")
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    st.divider()

    # ── Tweet más viral global ─────────────────────────────────────────────
    st.subheader("🔥 El tweet más viral del período")
    viral = df_prop.sort_values("like_count", ascending=False).iloc[0]
    with st.container(border=True):
        col_v1, col_v2 = st.columns([3, 1])
        with col_v1:
            foto_viral = get_foto(viral["candidato_nombre"])
            if foto_viral:
                st.image(foto_viral, width=52)
            st.markdown(
                f"**{viral['candidato_nombre']}** · {viral['candidato_pais']} · "
                f"{str(viral['created_at'])[:10]}"
            )
            st.markdown(f"> {str(viral['text'])[:300]}")
            if pd.notna(viral.get("tweet_url","")):
                st.markdown(f"[Ver en X ↗]({viral['tweet_url']})")
        with col_v2:
            st.metric("❤️ Likes",  fmt(viral["like_count"]))
            st.metric("🔁 RTs",    fmt(viral["retweet_count"]))
            st.metric("👁️ Views",  fmt(viral["view_count"]))

    st.divider()

    # ── Wordcloud ──────────────────────────────────────────────────────────
    st.subheader("¿De qué habla cada candidato?")
    nota("Términos más frecuentes en los tweets propios. Tamaño proporcional a la frecuencia.")
    candidatos_lista = sorted(df_prop["candidato_nombre"].unique().tolist())
    wc_cand  = st.selectbox("Selecciona un candidato:", candidatos_lista, key="wc_home")
    color_wc = COLORES_CANDIDATO.get(wc_cand, "#1F77B4")
    pais_wc  = df_prop[df_prop["candidato_nombre"] == wc_cand]["candidato_pais"].iloc[0]
    n_wc     = len(df_prop[df_prop["candidato_nombre"] == wc_cand])
    st.caption(f"{pais_wc} · {n_wc} tweets analizados")
    st.image(generar_wordcloud(wc_cand, color_wc), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS PAÍS
# ══════════════════════════════════════════════════════════════════════════════
with tab_chile:
    st.subheader("🇨🇱 Chile — Elecciones presidenciales 2025")
    render_contexto_pais("Chile")
    render_pais("Chile")

with tab_bolivia:
    st.subheader("🇧🇴 Bolivia — Elecciones presidenciales 2025")
    render_contexto_pais("Bolivia")
    render_pais("Bolivia")

with tab_ecuador:
    st.subheader("🇪🇨 Ecuador — Elecciones presidenciales 2025")
    render_contexto_pais("Ecuador")
    render_pais("Ecuador")
