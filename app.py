import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
import tempfile
import os
import random
import community as community_louvain

st.set_page_config(
    page_title="Análise de Redes Complexas - Marvel Universe",
    page_icon="🦸‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦸‍♂️ Análise de Redes do Universo Marvel")
st.markdown("### Rede de Relacionamentos entre Personagens Marvel")
st.markdown("""
Esta aplicação analisa a rede de relacionamentos entre personagens do Universo Marvel.
A rede representa co-aparições de personagens em quadrinhos.

**Dataset**: Marvel Universe Social Network
""")

st.sidebar.markdown("## 🦸‍♂️ Sobre o Dataset")
st.sidebar.info("""
**Marvel Universe Network**

Este dataset representa uma rede social onde:
- **Nós**: Personagens Marvel
- **Arestas**: Co-aparições em quadrinhos
- **Peso**: Força da relação (número de co-aparições)
- **Fonte**: Marvel Comics Database
""")

@st.cache_data
def load_marvel_data():
    nodes_url = "https://raw.githubusercontent.com/melaniewalsh/sample-social-network-datasets/refs/heads/master/sample-datasets/marvel/marvel-unimodal-nodes.csv"
    edges_url = "https://raw.githubusercontent.com/melaniewalsh/sample-social-network-datasets/refs/heads/master/sample-datasets/marvel/marvel-unimodal-edges.csv"
    
    try:
        # Carregar dados
        nodes_df = pd.read_csv(nodes_url)
        edges_df = pd.read_csv(edges_url)
        
        st.success(f"✅ Dados carregados: {len(nodes_df)} personagens, {len(edges_df)} relacionamentos")
        
        return nodes_df, edges_df
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar dataset Marvel: {e}")
        return None, None

@st.cache_data
def create_marvel_network(nodes_df, edges_df):
    G = nx.Graph()
    for idx, row in nodes_df.iterrows():
        G.add_node(row['Id'], label=row['Label'])
    for idx, row in edges_df.iterrows():
        if row['Source'] in G.nodes() and row['Target'] in G.nodes():
            G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    G_directed = nx.DiGraph()
    for idx, row in nodes_df.iterrows():
        G_directed.add_node(row['Id'], label=row['Label'])
    for idx, row in edges_df.iterrows():
        if row['Source'] in G_directed.nodes() and row['Target'] in G_directed.nodes():
            G_directed.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    return G, G_directed

@st.cache_data
def calculate_network_metrics(_G):
    n_nodes = _G.number_of_nodes()
    n_edges = _G.number_of_edges()
    if n_nodes == 0:
        return {
            'nodes': 0, 'edges': 0, 'density': 0, 'sparsity': 0, 'clustering': 0,
            'assortativity': 0, 'connected_components': 0, 'is_connected': False,
            'largest_component_size': 0, 'largest_component_ratio': 0,
            'diameter': 0, 'periphery': 0, 'avg_path_length': 0, 'avg_degree': 0
        }
    
    density = nx.density(_G)
    metrics = {
        'nodes': n_nodes, 'edges': n_edges, 'density': density, 'sparsity': 1 - density,
        'clustering': nx.average_clustering(_G),
        'connected_components': nx.number_connected_components(_G),
        'is_connected': nx.is_connected(_G),
        'avg_degree': sum(dict(_G.degree()).values()) / n_nodes
    }
    
    try:
        metrics['assortativity'] = nx.degree_assortativity_coefficient(_G)
    except:
        metrics['assortativity'] = 0

    components = list(nx.connected_components(_G))
    if components:
        largest_cc_nodes = max(components, key=len)
        metrics['largest_component_size'] = len(largest_cc_nodes)
        metrics['largest_component_ratio'] = len(largest_cc_nodes) / n_nodes

        G_largest = _G.subgraph(largest_cc_nodes)
        try:
            metrics['diameter'] = nx.diameter(G_largest)
            metrics['periphery'] = len(nx.periphery(G_largest))
            metrics['avg_path_length'] = nx.average_shortest_path_length(G_largest)
        except Exception:
            metrics['diameter'] = "N/A"
            metrics['periphery'] = "N/A"
            metrics['avg_path_length'] = "N/A"
    else:
        metrics['largest_component_size'] = 0
        metrics['largest_component_ratio'] = 0
        metrics['diameter'] = 0
        metrics['periphery'] = 0
        metrics['avg_path_length'] = 0
        
    return metrics

@st.cache_data
def calculate_centralities(_G, sample_size=500):
    if _G.number_of_nodes() == 0:
        return {}, _G
    
    if _G.number_of_nodes() > sample_size:
        degrees = dict(_G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:sample_size]
        nodes_sample = [node for node, degree in top_nodes]
        G_sample = _G.subgraph(nodes_sample).copy()
    else:
        G_sample = _G
    
    centralities = {
        'degree': dict(G_sample.degree()),
        'betweenness': nx.betweenness_centrality(G_sample),
        'closeness': nx.closeness_centrality(G_sample),
    }
    
    try:
        centralities['eigenvector'] = nx.eigenvector_centrality(G_sample, max_iter=1000)
    except:
        centralities['eigenvector'] = {n: G_sample.degree(n) for n in G_sample.nodes()}
    
    return centralities, G_sample

@st.cache_data
def get_character_names(_G, nodes_df):
    id_to_name = dict(zip(nodes_df['Id'], nodes_df['Label']))
    return id_to_name

def explain_metrics():
    with st.expander("📖 Explicação das Métricas Estruturais"):
        st.markdown("""
        **🔸 Densidade**: Proporção de relacionamentos existentes vs. possíveis (0-1). 
        Redes densas indicam universo altamente conectado.
        
        **🔸 Esparsidade**: Complemento da densidade (1 - densidade). 
        Indica quão "esparsa" é a rede de relacionamentos.
        
        **🔸 Assortatividade**: Tendência de personagens similares se relacionarem. 
        Positiva: personagens populares se relacionam entre si.
        
        **🔸 Coeficiente de Clustering**: Tendência de formar grupos fechados de personagens.
        Alto clustering indica formação de equipes/grupos.
        
        **🔸 Diâmetro**: Maior distância entre quaisquer dois personagens conectados.
        
        **🔸 Caminho Médio**: Distância média entre todos os pares de personagens.
        """)

def explain_centralities():
    with st.expander("📖 Explicação das Métricas de Centralidade"):
        st.markdown("""
        **🔸 Degree Centrality**: Número de co-aparições diretas. 
        Identifica personagens que aparecem com muitos outros.
        
        **🔸 Betweenness Centrality**: Frequência com que um personagem conecta outros. 
        Identifica personagens que fazem "ponte" entre grupos.
        
        **🔸 Closeness Centrality**: Quão próximo um personagem está de todos os outros. 
        Identifica personagens centrais no universo.
        
        **🔸 Eigenvector Centrality**: Considera não apenas número de conexões, mas importância dos parceiros. 
        Identifica personagens conectados a outros personagens importantes.
        """)

with st.spinner("🔄 Carregando dados do Universo Marvel..."):
    nodes_df, edges_df = load_marvel_data()

if nodes_df is None or edges_df is None:
    st.error("❌ Não foi possível carregar os dados Marvel.")
    st.stop()

# Criar rede
with st.spinner("🔄 Processando rede Marvel..."):
    G, G_directed = create_marvel_network(nodes_df, edges_df)
    metrics = calculate_network_metrics(G)
    centralities, G_sample = calculate_centralities(G)
    id_to_name = get_character_names(G, nodes_df)

if metrics['nodes'] == 0:
    st.error("❌ Erro ao processar a rede Marvel.")
    st.stop()

st.sidebar.markdown("## 🎛️ Controles da Visualização")

subset_option = st.sidebar.selectbox(
    "🔍 Selecionar Subconjunto:",
    ["Rede Completa", "Componente Gigante", "Alto Grau", "Heróis Principais"]
)

if subset_option == "Alto Grau":
    max_degree = max([G.degree(n) for n in G.nodes()]) if G.number_of_nodes() > 0 else 20
    min_degree_threshold = st.sidebar.slider(
        "Co-aparições mínimas:", 
        1, min(50, max_degree), 
        min(5, max_degree//4)
    )
else:
    min_degree_threshold = 1

layout_option = st.sidebar.selectbox(
    "🎨 Layout da Rede:",
    ["spring", "circular", "kamada_kawai", "shell"]
)

centrality_filter = st.sidebar.selectbox(
    "⭐ Destacar por Centralidade:",
    ["Nenhum", "Degree", "Betweenness", "Closeness", "Eigenvector"]
)

top_k = st.sidebar.slider("🔝 Top K personagens destacados:", 5, 50, 15)

max_nodes_viz = st.sidebar.slider("📊 Máximo de nós na visualização:", 50, 500, 200)

G_filtered = G.copy()

if subset_option == "Componente Gigante":
    if nx.number_connected_components(G_filtered) > 1:
        largest_cc = max(nx.connected_components(G_filtered), key=len)
        G_filtered = G_filtered.subgraph(largest_cc).copy()
elif subset_option == "Alto Grau":
    nodes_to_remove = [n for n in G_filtered.nodes() if G_filtered.degree(n) < min_degree_threshold]
    G_filtered.remove_nodes_from(nodes_to_remove)
elif subset_option == "Heróis Principais":
    degrees = dict(G_filtered.degree())
    top_heroes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes_viz]
    nodes_to_keep = [node for node, degree in top_heroes]
    G_filtered = G_filtered.subgraph(nodes_to_keep).copy()

if G_filtered.number_of_nodes() > max_nodes_viz:
    degrees = dict(G_filtered.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes_viz]
    nodes_sample = [node for node, degree in top_nodes]
    G_filtered = G_filtered.subgraph(nodes_sample).copy()

st.markdown("## 1. 🦸‍♂️ Visualização da Rede Marvel")

col1, col2 = st.columns([3, 1])

with col1:
    if G_filtered.number_of_nodes() == 0:
        st.warning("⚠️ Nenhum personagem encontrado com os filtros aplicados.")
    else:
        net = Network(height="600px", width="100%", bgcolor="white", font_color="black")
        
        try:
            if layout_option == "spring":
                pos = nx.spring_layout(G_filtered, k=1, iterations=50)
            elif layout_option == "circular":
                pos = nx.circular_layout(G_filtered)
            elif layout_option == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G_filtered)
            elif layout_option == "shell":
                pos = nx.shell_layout(G_filtered)
        except:
            pos = nx.spring_layout(G_filtered)
        
        top_nodes = []
        if centrality_filter != "Nenhum" and G_filtered.number_of_nodes() > 0:
            if centrality_filter == "Degree":
                cent_dict = {n: G_filtered.degree(n) for n in G_filtered.nodes()}
            else:
                try:
                    if centrality_filter == "Betweenness":
                        cent_dict = nx.betweenness_centrality(G_filtered)
                    elif centrality_filter == "Closeness":
                        cent_dict = nx.closeness_centrality(G_filtered)
                    elif centrality_filter == "Eigenvector":
                        cent_dict = nx.eigenvector_centrality(G_filtered, max_iter=1000)
                except:
                    cent_dict = {n: G_filtered.degree(n) for n in G_filtered.nodes()}
            
            top_nodes = sorted(cent_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            top_nodes = [n[0] for n in top_nodes]
        
        for node in G_filtered.nodes():
            degree = G_filtered.degree(node)
            char_name = id_to_name.get(node, f"Character {node}")
            
            size = 10 + degree * 0.8
            color = '#3498db'
            
            if node in top_nodes:
                color = '#e74c3c'
                size += 8
            
            if any(keyword in char_name.lower() for keyword in ['spider', 'captain', 'iron', 'wolverine', 'thor', 'hulk']):
                color = '#f39c12'
                size += 3
            
            tooltip = f"<b>{char_name}</b><br>Co-aparições: {degree}"
            if G_filtered.has_edge(node, node):
                weight = G_filtered[node][node].get('weight', 1)
                tooltip += f"<br>Peso da conexão: {weight}"
            
            net.add_node(
                node,
                label=char_name if len(char_name) < 15 else char_name[:12] + "...",
                color=color,
                size=size,
                x=pos[node][0] * 400,
                y=pos[node][1] * 400,
                title=tooltip
            )
        
        for edge in G_filtered.edges(data=True):
            weight = edge[2].get('weight', 1)
            edge_width = min(1 + weight * 0.1, 5)
            
            net.add_edge(
                edge[0], 
                edge[1], 
                color='#95a5a6', 
                width=edge_width,
                title=f"Co-aparições: {weight}"
            )
        
        net.set_options("""
        var options = {
          "physics": {
            "enabled": false,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -3000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            }
          }
        }
        """)
    
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            tmp_path = tmp.name
        
        net.save_graph(tmp_path)
        
        try:
            with open(tmp_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=650)
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

with col2:
    st.markdown("#### 📊 Info da Visualização")
    st.metric("Personagens Exibidos", f"{G_filtered.number_of_nodes():,}")
    st.metric("Relacionamentos Exibidos", f"{G_filtered.number_of_edges():,}")
    st.metric("Subconjunto", subset_option)
    
    st.markdown("#### 🎨 Legenda")
    st.markdown("🔵 Personagens normais")
    st.markdown("🔴 Top K destacados")
    st.markdown("🟠 Heróis famosos")
    
    if centrality_filter != "Nenhum":
        st.markdown(f"**Destacando por**: {centrality_filter}")

st.markdown("## 2. 📏 Métricas Estruturais")
explain_metrics()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🦸‍♂️ Personagens", f"{metrics['nodes']:,}")
    st.metric("🔗 Relacionamentos", f"{metrics['edges']:,}")

with col2:
    st.metric("📊 Densidade", f"{metrics['density']:.6f}")
    st.metric("🕳️ Esparsidade", f"{metrics['sparsity']:.6f}")

with col3:
    st.metric("🔄 Assortatividade", f"{metrics['assortativity']:.3f}")
    st.metric("🌐 Clustering Global", f"{metrics['clustering']:.3f}")

with col4:
    st.metric("🧩 Componentes", f"{metrics['connected_components']:,}")
    st.metric("📏 Diâmetro", f"{metrics['diameter']}")
    st.metric("🌍 Nós na Periferia", f"{metrics.get('periphery', 'N/A')}")

st.markdown("#### 🧩 Análise de Conectividade")
col1, col2 = st.columns(2)

with col1:
    st.metric("🏔️ Maior Componente", f"{metrics['largest_component_size']:,} personagens")
    st.metric("📊 % do Maior Componente", f"{metrics['largest_component_ratio']:.1%}")

with col2:
    connectivity_status = "✅ Conectada" if metrics['is_connected'] else "❌ Desconectada"
    st.markdown(f"**Status da Rede**: {connectivity_status}")
    st.metric("📊 Grau Médio", f"{metrics['avg_degree']:.2f}")
    if metrics['avg_path_length'] > 0:
        st.metric("🛤️ Caminho Médio", f"{metrics['avg_path_length']:.2f}")

# Distribuições de grau
st.markdown("## 3. 📈 Distribuições de Co-aparições")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Histograma de Co-aparições")
    degrees = [G.degree(n) for n in G.nodes()]
    
    fig_hist = px.histogram(
        x=degrees,
        nbins=min(50, len(set(degrees))),
        title="Distribuição de Co-aparições",
        labels={'x': 'Número de Co-aparições', 'y': 'Frequência'},
        color_discrete_sequence=['#3498db']
    )
    fig_hist.update_layout(showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("#### 📊 Estatísticas")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Média", f"{np.mean(degrees):.2f}")
        st.metric("Mediana", f"{np.median(degrees):.0f}")
    with col_b:
        st.metric("Máximo", f"{max(degrees)}")
        st.metric("Mínimo", f"{min(degrees)}")

with col2:
    st.subheader("Top 15 Personagens Mais Conectados")
    
    top_degree_nodes = sorted([(n, G.degree(n)) for n in G.nodes()], 
                              key=lambda x: x[1], reverse=True)[:15]
    
    char_names = []
    degrees_list = []
    
    for node, degree in top_degree_nodes:
        char_name = id_to_name.get(node, f"Character {node}")
        char_names.append(char_name[:25] + "..." if len(char_name) > 25 else char_name)
        degrees_list.append(degree)
    
    fig_top = px.bar(
        x=degrees_list,
        y=char_names,
        orientation='h',
        title="Personagens com Mais Co-aparições",
        labels={'x': 'Co-aparições', 'y': 'Personagem'},
        color_discrete_sequence=['#e74c3c']
    )
    fig_top.update_layout(height=500)
    st.plotly_chart(fig_top, use_container_width=True)

st.markdown("## 4. ⭐ Centralidade dos Personagens")
explain_centralities()

if G_sample.number_of_nodes() > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlação entre Centralidades")
        
        sample_ids = list(centralities['degree'].keys())
        sample_names = [id_to_name.get(node_id, f"Character {node_id}") for node_id in sample_ids]
        
        cent_df = pd.DataFrame({
            'Node': sample_ids,
            'Name': sample_names,
            'Degree': list(centralities['degree'].values()),
            'Betweenness': list(centralities['betweenness'].values()),
            'Closeness': list(centralities['closeness'].values()),
            'Eigenvector': list(centralities['eigenvector'].values())
        })
        
        corr_cols = ['Degree', 'Betweenness', 'Closeness', 'Eigenvector']
        corr_matrix = cent_df[corr_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlação entre Métricas de Centralidade",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.subheader("Rankings por Centralidade")
        
        selected_centrality = st.selectbox(
            "Selecione a centralidade para ranking:",
            ['Degree', 'Betweenness', 'Closeness', 'Eigenvector']
        )
        
        top_central = cent_df.nlargest(15, selected_centrality)[['Name', selected_centrality]]
        
        st.markdown(f"#### 🏆 Top 15 - {selected_centrality} Centrality")
        
        for idx, (_, row) in enumerate(top_central.iterrows(), 1):
            name = row['Name']
            value = row[selected_centrality]
            if selected_centrality == 'Degree':
                st.write(f"{idx}. **{name}**: {int(value)} co-aparições")
            else:
                st.write(f"{idx}. **{name}**: {value:.4f}")
    
    st.subheader("Personagens Mais Importantes por Centralidade")
    
    fig_centralities = px.scatter_matrix(
        cent_df[corr_cols], 
        title="Comparação entre Métricas de Centralidade",
        height=600
    )
    st.plotly_chart(fig_centralities, use_container_width=True)

    st.subheader("🏆 Hall da Fama Marvel")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🔗 Mais Conectados**")
        top_degree = cent_df.nlargest(5, 'Degree')
        for idx, (_, row) in enumerate(top_degree.iterrows(), 1):
            st.write(f"{idx}. {row['Name']}")
    
    with col2:
        st.markdown("**🌉 Maiores Pontes**")
        top_between = cent_df.nlargest(5, 'Betweenness')
        for idx, (_, row) in enumerate(top_between.iterrows(), 1):
            st.write(f"{idx}. {row['Name']}")
    
    with col3:
        st.markdown("**🎯 Mais Centrais**")
        top_close = cent_df.nlargest(5, 'Closeness')
        for idx, (_, row) in enumerate(top_close.iterrows(), 1):
            st.write(f"{idx}. {row['Name']}")
    
    with col4:
        st.markdown("**⭐ Mais Influentes**")
        top_eigen = cent_df.nlargest(5, 'Eigenvector')
        for idx, (_, row) in enumerate(top_eigen.iterrows(), 1):
            st.write(f"{idx}. {row['Name']}")

else:
    st.warning("⚠️ Não foi possível calcular centralidades para este subconjunto.")

# Análise de componentes
st.markdown("## 5. 🧩 Análise de Grupos de Personagens")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribuição de Grupos")
    components = list(nx.connected_components(G))
    component_sizes = [len(comp) for comp in components]
    
    fig_comp = px.histogram(
        x=component_sizes,
        nbins=min(30, len(component_sizes)),
        title="Distribuição de Tamanhos dos Grupos",
        labels={'x': 'Tamanho do Grupo', 'y': 'Frequência'},
        color_discrete_sequence=['#9b59b6']
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("#### 📊 Estatísticas dos Grupos")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total de Grupos", f"{len(components):,}")
        if component_sizes:
            st.metric("Maior Grupo", f"{max(component_sizes):,} personagens")
    with col_b:
        if component_sizes:
            st.metric("Menor Grupo", f"{min(component_sizes)} personagens")
            st.metric("Tamanho Médio", f"{np.mean(component_sizes):.2f}")

with col2:
    st.subheader("Maiores Grupos Marvel")
    if component_sizes:
        largest_components = sorted(component_sizes, reverse=True)[:10]
        
        fig_bar = px.bar(
            x=range(1, len(largest_components) + 1),
            y=largest_components,
            title="10 Maiores Grupos de Personagens",
            labels={'x': 'Ranking', 'y': 'Tamanho'},
            color_discrete_sequence=['#f39c12']
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("#### 🏝️ Personagens Isolados")
        col_a, col_b = st.columns(2)
        isolated_nodes = len([comp for comp in components if len(comp) == 1])
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Personagens Isolados", f"{isolated_nodes:,}")
        with col_b:
            if G.number_of_nodes() > 0:
                isolation_pct = isolated_nodes/G.number_of_nodes()*100
                st.metric("% Isolados", f"{isolation_pct:.1f}%")
st.markdown("## 6. Análises Adicionais Requeridas")

tab1, tab2, tab3, tab4 = st.tabs([
    "Detecção de Comunidades (Louvain)",
    "Componentes Conectados",
    "Clustering Local",
    "Matriz de Adjacência"
])

with tab1:
    st.subheader("Communities Detection with Louvain Algorithm")
    st.info("""
    O algoritmo de Louvain detecta **comunidades** (grupos de nós densamente conectados) na rede. Ele maximiza a **modularidade**, uma medida da qualidade da divisão da rede em comunidades.
    """)
    
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        partition = community_louvain.best_partition(G, weight='weight')
        modularity = community_louvain.modularity(partition, G, weight='weight')
        num_communities = len(set(partition.values()))

        col1, col2 = st.columns(2)
        col1.metric("Número de Comunidades Detectadas", f"{num_communities:,}")
        col2.metric("Modularidade da Partição", f"{modularity:.4f}")

        # Organizar comunidades por tamanho
        communities_dict = {}
        for node, community in partition.items():
            if community not in communities_dict:
                communities_dict[community] = []
            communities_dict[community].append(node)
        
        # Ordenar comunidades por tamanho (maior primeiro)
        sorted_communities = sorted(communities_dict.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Estatísticas das comunidades
        community_sizes = [len(members) for _, members in sorted_communities]
        
        st.markdown("#### 📊 Estatísticas das Comunidades")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Maior Comunidade", f"{max(community_sizes)} personagens")
        with col2:
            st.metric("Menor Comunidade", f"{min(community_sizes)} personagens")
        with col3:
            st.metric("Tamanho Médio", f"{np.mean(community_sizes):.1f}")
        with col4:
            st.metric("Tamanho Mediano", f"{np.median(community_sizes):.0f}")
        
        # Gráfico de distribuição de tamanhos
        fig_comm_sizes = px.histogram(
            x=community_sizes,
            nbins=min(20, len(community_sizes)),
            title="Distribuição de Tamanhos das Comunidades",
            labels={'x': 'Tamanho da Comunidade', 'y': 'Frequência'},
            color_discrete_sequence=['#8e44ad']
        )
        st.plotly_chart(fig_comm_sizes, use_container_width=True)
        
        # Mostrar as maiores comunidades
        st.markdown("#### 🏆 Maiores Comunidades Detectadas")
        
        num_communities_to_show = min(10, len(sorted_communities))
        
        for i, (community_id, members) in enumerate(sorted_communities[:num_communities_to_show]):
            with st.expander(f"🔸 Comunidade {community_id + 1} - {len(members)} personagens", expanded=(i < 3)):
                
                # Converter IDs para nomes
                member_names = []
                for member_id in members:
                    name = id_to_name.get(member_id, f"Character {member_id}")
                    member_names.append(name)
                
                # Ordenar nomes alfabeticamente
                member_names.sort()
                
                # Mostrar estatísticas da comunidade
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Tamanho:** {len(members)} personagens")
                    
                    # Calcular densidade interna da comunidade
                    subgraph = G.subgraph(members)
                    internal_density = nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0
                    st.markdown(f"**Densidade Interna:** {internal_density:.3f}")
                
                with col2:
                    # Encontrar o personagem mais conectado dentro da comunidade
                    degrees_in_community = {node: G.degree(node) for node in members}
                    most_connected = max(degrees_in_community, key=degrees_in_community.get)
                    most_connected_name = id_to_name.get(most_connected, f"Character {most_connected}")
                    st.markdown(f"**Personagem Central:** {most_connected_name}")
                    st.markdown(f"**Grau:** {degrees_in_community[most_connected]}")
                
                # Mostrar membros da comunidade
                st.markdown("**Membros da Comunidade:**")
                
                # Dividir em colunas para melhor visualização
                cols = st.columns(3)
                for idx, name in enumerate(member_names):
                    col_idx = idx % 3
                    with cols[col_idx]:
                        # Destacar o personagem mais conectado
                        if name == most_connected_name:
                            st.markdown(f"⭐ **{name}**")
                        else:
                            st.markdown(f"• {name}")
        
        # Visualização das comunidades como gráfico de barras
        st.markdown("#### 📈 Ranking das Comunidades por Tamanho")
        
        top_communities = sorted_communities[:15]  # Top 15 comunidades
        community_labels = [f"Comunidade {comm_id + 1}" for comm_id, _ in top_communities]
        community_sizes_top = [len(members) for _, members in top_communities]
        
        fig_ranking = px.bar(
            x=community_sizes_top,
            y=community_labels,
            orientation='h',
            title="Top 15 Comunidades por Número de Personagens",
            labels={'x': 'Número de Personagens', 'y': 'Comunidade'},
            color_discrete_sequence=['#e74c3c']
        )
        fig_ranking.update_layout(height=500)
        st.plotly_chart(fig_ranking, use_container_width=True)
        
        # Análise de modularidade
        st.markdown("#### 🔍 Análise de Qualidade da Partição")
        
        if modularity > 0.3:
            quality_status = "🟢 Excelente"
            quality_desc = "A partição em comunidades é muito boa. As comunidades são bem definidas e densas internamente."
        elif modularity > 0.2:
            quality_status = "🟡 Boa"
            quality_desc = "A partição é razoável. Há uma estrutura de comunidades clara, mas pode haver sobreposições."
        else:
            quality_status = "🔴 Fraca"
            quality_desc = "A partição em comunidades é fraca. A rede pode não ter uma estrutura de comunidades bem definida."
        
        st.markdown(f"**Status da Modularidade:** {quality_status}")
        st.markdown(f"**Interpretação:** {quality_desc}")
        
        # Salvar informações das comunidades para uso posterior
        nx.set_node_attributes(G, partition, 'community')
        st.success("✅ As comunidades foram detectadas e analisadas com sucesso!")
        
        # Informações adicionais
        with st.expander("ℹ️ Mais Informações sobre o Algoritmo de Louvain"):
            st.markdown("""
            **Como funciona o Algoritmo de Louvain:**
            
            1. **Inicialização:** Cada nó começa como sua própria comunidade
            2. **Otimização Local:** Para cada nó, testa mover para comunidades vizinhas
            3. **Agregação:** Cria um novo grafo onde cada comunidade vira um super-nó
            4. **Repetição:** Repete até não haver mais melhoria na modularidade
            
            **Vantagens:**
            - Muito rápido para redes grandes
            - Encontra comunidades hierárquicas
            - Maximiza a modularidade
            
            **Limitações:**
            - Pode encontrar diferentes partições em execuções diferentes
            - Favorece comunidades de tamanhos similares
            - Não detecta comunidades sobrepostas
            """)
    
    else:
        st.warning("Não é possível detectar comunidades em um grafo vazio ou sem arestas.")

with tab2:
    st.subheader("Componentes Fortemente e Fracamente Conectados")
    st.info("""
    Esta análise requer um **grafo dirigido**.
    - **Componentes Fortemente Conectados (SCC):** Grupos onde todo personagem pode alcançar qualquer outro do mesmo grupo. São núcleos muito coesos.
    - **Componentes Fracamente Conectados (WCC):** Grupos onde há um caminho entre quaisquer dois personagens, ignorando a direção das arestas.
    """)
    
    # Usando o grafo dirigido criado no Passo 3
    scc = list(nx.strongly_connected_components(G_directed))
    wcc = list(nx.weakly_connected_components(G_directed))
    
    col1, col2 = st.columns(2)
    col1.metric("Nº de Componentes Fortemente Conectados", f"{len(scc):,}")
    col2.metric("Nº de Componentes Fracamente Conectados", f"{len(wcc):,}")

with tab3:
    st.subheader("Coeficiente de Clustering Local")
    st.info("Mede quão próximos os vizinhos de um nó estão de serem um 'clique' (grupo completo).")
    
    # Pegar uma amostra de personagens para não sobrecarregar o selectbox
    sample_nodes = list(G.nodes())
    if len(sample_nodes) > 200:
        degrees = dict(G.degree())
        sample_nodes = sorted(degrees, key=degrees.get, reverse=True)[:200]

    char_names_sample = sorted([id_to_name.get(node) for node in sample_nodes if id_to_name.get(node)])
    
    char_choice = st.selectbox("Escolha um personagem para analisar:", char_names_sample, index=0)
    
    if char_choice:
        node_id = [node for node, name in id_to_name.items() if name == char_choice][0]
        local_clustering = nx.clustering(G, node_id)
        st.metric(f"Clustering Local para {char_choice}", f"{local_clustering:.4f}")

with tab4:
    st.subheader("Matriz de Adjacência")
    st.info("""
    Representação da rede onde `A[i, j] = 1` se os personagens `i` e `j` estão conectados. Abaixo, uma amostra da matriz para os 15 primeiros personagens.
    """)

    sample_nodes_adj = list(G.nodes())[:15]
    adj_matrix = nx.to_pandas_adjacency(G, nodelist=sample_nodes_adj)
    st.dataframe(adj_matrix)

st.markdown("---")
st.markdown("## 🔬 Análise Crítica da Rede Marvel")

with st.expander("Clique aqui para ver a análise crítica detalhada", expanded=False):

    st.info("""
    Nesta seção, vamos além das métricas básicas para entender a natureza fundamental da rede da Marvel,
    comparando-a com modelos teóricos e testando suas propriedades avançadas, como resiliência e potencial de crescimento.
    """)

    tab_modelos, tab_resiliencia, tab_links = st.tabs([
        "Comparação com Modelos Teóricos",
        "Análise de Resiliência e Robustez",
        "Predição de Links (Link Prediction)"
    ])

    with tab_modelos:
        st.subheader("A Rede Marvel é um Mundo Pequeno ou Livre de Escala?")
        st.markdown("""
        Redes reais raramente são completamente aleatórias. Elas geralmente seguem padrões, como os de **Mundo Pequeno** (alta clusterização e caminhos curtos, como no modelo de Watts-Strogatz) ou **Livre de Escala** (distribuição de grau seguindo uma Lei de Potência, com hubs, como no modelo Barabási-Albert).
        """)

        actual_clustering = nx.average_clustering(G)
        
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        if n_nodes > 0 and n_edges > 0:
            p_random = (2 * n_edges) / (n_nodes * (n_nodes - 1))
            
            expected_clustering_random = p_random
            
            actual_path_length = metrics.get('avg_path_length', 'N/A')
            
            if p_random > 0:
                expected_path_random = np.log(n_nodes) / np.log(n_nodes * p_random)
            else:
                expected_path_random = "N/A"
            
            st.markdown("#### 📊 Comparação com Rede Aleatória Equivalente")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Coeficiente de Clustering:**")
                st.metric("Rede Marvel", f"{actual_clustering:.4f}")
                st.metric("Rede Aleatória (esperado)", f"{expected_clustering_random:.4f}")
                
                if actual_clustering > expected_clustering_random * 2:
                    st.success("✅ **Mundo Pequeno**: Clustering muito maior que o esperado!")
                elif actual_clustering > expected_clustering_random:
                    st.info("🔸 Clustering moderadamente maior que o esperado")
                else:
                    st.warning("⚠️ Clustering similar ao de rede aleatória")
            
            with col2:
                st.markdown("**Caminho Médio:**")
                if actual_path_length != "N/A":
                    st.metric("Rede Marvel", f"{actual_path_length:.2f}")
                else:
                    st.metric("Rede Marvel", "N/A")
                
                if expected_path_random != "N/A":
                    st.metric("Rede Aleatória (esperado)", f"{expected_path_random:.2f}")
                else:
                    st.metric("Rede Aleatória (esperado)", "N/A")
                
                if actual_path_length != "N/A" and expected_path_random != "N/A":
                    ratio = actual_path_length / expected_path_random
                    if ratio < 2:
                        st.success("✅ **Mundo Pequeno**: Caminho médio similar ao aleatório!")
                    else:
                        st.info("🔸 Caminho médio um pouco maior que o esperado")

        st.markdown("#### 🔍 Análise da Distribuição de Grau (Lei de Potência)")
        
        degrees = [G.degree(n) for n in G.nodes()]
        
        if degrees:
            degree_counts = pd.Series(degrees).value_counts().sort_index()
            
            degree_values = degree_counts.index.values
            frequencies = degree_counts.values
            probabilities = frequencies / sum(frequencies)
            
            valid_mask = (degree_values > 0) & (probabilities > 0)
            log_degrees = np.log10(degree_values[valid_mask])
            log_probs = np.log10(probabilities[valid_mask])
            
            fig_log = go.Figure()
            fig_log.add_trace(go.Scatter(
                x=degree_values[valid_mask], 
                y=probabilities[valid_mask], 
                mode='markers',
                name='Distribuição Empírica',
                marker=dict(size=8, color='#3498db')
            ))
            
            if len(log_degrees) > 1:
                slope, intercept = np.polyfit(log_degrees, log_probs, 1)
                fitted_line = 10**(slope * log_degrees + intercept)
                
                fig_log.add_trace(go.Scatter(
                    x=degree_values[valid_mask],
                    y=fitted_line,
                    mode='lines',
                    name=f'Ajuste Linear (γ ≈ {-slope:.2f})',
                    line=dict(color='red', dash='dash')
                ))
            
            fig_log.update_layout(
                title="Distribuição de Grau em Escala Log-Log",
                xaxis_title="Grau (k)",
                yaxis_title="Probabilidade P(k)",
                xaxis_type="log",
                yaxis_type="log",
                showlegend=True
            )
            st.plotly_chart(fig_log, use_container_width=True)
            
            st.markdown("#### 📈 Estatísticas da Distribuição de Grau")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Grau Médio", f"{np.mean(degrees):.2f}")
                st.metric("Grau Mediano", f"{np.median(degrees):.0f}")
            
            with col2:
                st.metric("Grau Máximo", f"{max(degrees)}")
                st.metric("Grau Mínimo", f"{min(degrees)}")
            
            with col3:
                st.metric("Desvio Padrão", f"{np.std(degrees):.2f}")
                skewness = pd.Series(degrees).skew()
                st.metric("Assimetria", f"{skewness:.2f}")
            
            if len(log_degrees) > 1:
                correlation = np.corrcoef(log_degrees, log_probs)[0, 1]
                r_squared = correlation ** 2
                
                st.markdown("#### 🎯 Evidência de Lei de Potência")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Expoente γ (gamma)", f"{-slope:.3f}")
                    st.metric("R² do Ajuste Log-Log", f"{r_squared:.3f}")
                
                with col2:
                    if r_squared > 0.8:
                        st.success("✅ **Forte evidência** de Lei de Potência")
                    elif r_squared > 0.6:
                        st.info("🔸 **Evidência moderada** de Lei de Potência")
                    else:
                        st.warning("⚠️ **Fraca evidência** de Lei de Potência")
                    
                    if 2 <= -slope <= 3:
                        st.info("🔸 Expoente típico de redes livre de escala")
                    else:
                        st.info("🔸 Expoente fora do range típico (2-3)")

        st.markdown("#### 🏁 Conclusão da Análise de Modelos")
        
        is_small_world = False
        is_scale_free = False
        
        if actual_clustering > expected_clustering_random * 2:
            is_small_world = True
        
        if len(log_degrees) > 1 and r_squared > 0.6:
            is_scale_free = True
        
        if is_small_world and is_scale_free:
            conclusion = "🎯 **Rede Livre de Escala com Propriedades de Mundo Pequeno**"
            explanation = """
            A rede Marvel exibe características de ambos os modelos:
            - **Mundo Pequeno**: Alta clusterização (formação de grupos como Vingadores, X-Men)
            - **Livre de Escala**: Distribuição de grau seguindo lei de potência (poucos hubs com muitas conexões)
            
            Isso é comum em redes sociais reais, onde há tanto grupos locais quanto conectores globais.
            """
        elif is_scale_free:
            conclusion = "📈 **Rede Livre de Escala**"
            explanation = """
            A rede Marvel segue primariamente um modelo livre de escala, com alguns personagens 
            (hubs) tendo um número desproporcionalmente grande de conexões. Isso reflete o processo 
            de "conexão preferencial" onde novos personagens tendem a se conectar com os já famosos.
            """
        elif is_small_world:
            conclusion = "🌐 **Rede de Mundo Pequeno**"
            explanation = """
            A rede Marvel mostra principalmente características de mundo pequeno, com alta clusterização 
            local (formação de grupos) mas caminhos curtos entre qualquer par de personagens.
            """
        else:
            conclusion = "❓ **Rede com Características Mistas**"
            explanation = """
            A rede Marvel não se encaixa perfeitamente em nenhum dos modelos clássicos, 
            apresentando características únicas que podem refletir a natureza específica 
            do universo narrativo da Marvel.
            """
        
        st.markdown(f"**{conclusion}**")
        st.markdown(explanation)


    with tab_links:
        st.subheader("Predição de Links: Quem Deveria Formar uma Equipe?")
        st.markdown("""
        Podemos usar a topologia da rede para prever quais personagens que **ainda não interagiram** têm a maior probabilidade de colaborar no futuro. Usaremos o **Índice Adamic-Adar**, que dá mais peso a vizinhos em comum que são menos conectados.

        Em outras palavras, se dois personagens (A e B) são ambos amigos de um personagem "exclusivo" (C), a chance de A e B se conectarem é maior do que se fossem amigos de um "hub" como o Capitão América.
        """)

        if st.button("Calcular Principais Alianças Futuras"):
            with st.spinner("Analisando potenciais novas amizades..."):
                main_nodes = max(nx.connected_components(G), key=len)
                G_main = G.subgraph(main_nodes)

                possible_edges = list(nx.non_edges(G_main))
                predictions = list(nx.adamic_adar_index(G_main, possible_edges))
                
                pred_df = pd.DataFrame(predictions, columns=['Nó A', 'Nó B', 'Score Adamic-Adar'])
                pred_df['Personagem A'] = pred_df['Nó A'].map(id_to_name)
                pred_df['Personagem B'] = pred_df['Nó B'].map(id_to_name)

                st.write("Top 15 Parcerias Mais Prováveis (que ainda não aconteceram):")
                st.dataframe(
                    pred_df[['Personagem A', 'Personagem B', 'Score Adamic-Adar']]
                    .nlargest(15, 'Score Adamic-Adar')
                    .reset_index(drop=True)
                )

                st.markdown("""
                **Conclusão da Predição de Links:**
                Os resultados acima mostram pares de personagens que, com base em seus círculos sociais compartilhados, têm um alto potencial de conexão. Esta técnica é usada em redes sociais como LinkedIn ("pessoas que você talvez conheça") e pode ter aplicações interessantes para prever dinâmicas narrativas em universos ficcionais.
                """)

st.markdown("---")
st.markdown("## 📚 Informações do Projeto")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🛠️ Tecnologias Utilizadas:**
    - Streamlit (Interface Web)
    - NetworkX (Análise de Redes)
    - Pyvis (Visualização Interativa)
    - Plotly (Gráficos)
    - Pandas & NumPy (Processamento)
    """)

with col2:
    st.markdown("""
    **📊 Dataset:**
    - Marvel Universe Social Network
    - Nós: Personagens Marvel
    - Arestas: Co-aparições em quadrinhos
    - Fonte: Marvel Comics Database
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>🦸‍♂️ Análise de Redes do Universo Marvel</strong></p>
    <p>Desenvolvido com Streamlit • NetworkX • Pyvis</p>
    <p>Hebert França</p>
</div>
""", unsafe_allow_html=True)