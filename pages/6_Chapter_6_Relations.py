import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from collections import deque

# ==========================================
# 1. 页面配置与样式
# ==========================================
st.set_page_config(page_title="Ch 6: Relations & Graphs", layout="wide")

st.markdown("""
<style>
    .math-tag { background-color: #e3f2fd; color: #0d47a1; padding: 4px 8px; border-radius: 5px; font-weight: bold; }
    .db-tag { background-color: #fce4ec; color: #880e4f; padding: 4px 8px; border-radius: 5px; font-weight: bold; }
    .highlight-box { background-color: #f0f2f6; border-left: 5px solid #4caf50; padding: 15px; margin: 10px 0; border-radius: 5px; }
    h3 { color: #1f77b4; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心算法库 (Backend Logic)
# ==========================================

def parse_set_input(input_str):
    try: return sorted(list(set([int(x.strip()) for x in input_str.split(',') if x.strip()])))
    except: return [1, 2, 3, 4]

def generate_relation_data(set_a, set_b, rule):
    relation = []
    for a in set_a:
        for b in set_b:
            is_related = False
            if rule == "Less Than (a < b)": is_related = (a < b)
            elif rule == "Equal (a = b)": is_related = (a == b)
            elif rule == "Divides (a | b)": is_related = (a != 0 and b % a == 0)
            if is_related: relation.append((a, b))
    return relation

def check_properties(A, R_list):
    R_set = set(R_list)
    props = {}
    props['Reflexive'] = all((a,a) in R_set for a in A)
    props['Symmetric'] = all((b,a) in R_set for (a,b) in R_list)
    props['Anti-symmetric'] = True
    for (a,b) in R_list:
        if a != b and (b,a) in R_set: props['Anti-symmetric'] = False; break
    props['Transitive'] = True
    for (a,b) in R_list:
        for (c,d) in R_list:
            if b == c and (a,d) not in R_set: props['Transitive'] = False; break
        if not props['Transitive']: break
    return props

def get_matrix(nodes, edges):
    size = len(nodes)
    matrix = np.zeros((size, size), dtype=int)
    idx_map = {val: i for i, val in enumerate(nodes)}
    for u, v in edges:
        # 兼容数字节点和字符串节点
        if u in idx_map and v in idx_map:
            matrix[idx_map[u]][idx_map[v]] = 1
    return matrix, idx_map

def matrix_power(matrix, k):
    """计算布尔矩阵的 k 次幂 (用于模拟路径查找)"""
    res = matrix
    for _ in range(k-1):
        res = np.dot(res, matrix)
    return (res > 0).astype(int) # 保持为 0/1 矩阵

def topological_sort(nodes, edges):
    """简化的拓扑排序算法 (Kahn's Algorithm)"""
    in_degree = {node: 0 for node in nodes}
    for u, v in edges:
        if v in in_degree: in_degree[v] += 1
    
    queue = deque([n for n in nodes if in_degree[n] == 0])
    sorted_list = []
    
    while queue:
        u = queue.popleft()
        sorted_list.append(u)
        for src, dest in edges:
            if src == u and dest in in_degree:
                in_degree[dest] -= 1
                if in_degree[dest] == 0:
                    queue.append(dest)
    
    if len(sorted_list) != len(nodes):
        return None # 有环
    return sorted_list

# ==========================================
# 3. 模块渲染函数 (Frontend View)
# ==========================================

def render_overview():
    st.header("Chapter 6: Relations as Structures")
    st.markdown("""
    Based on our meetings, this tool connects **Discrete Math** to **Computer Science** in 4 Key Areas:
    
    1.  **Sets to SQL (6.1, 6.2, 6.10)**: Understanding that Relations are just Database Tables.
    2.  **Modeling Networks (6.3, 6.6)**: Using Graphs and Matrices to represent connections.
    3.  **Operations (6.4, 6.5)**: Composition and Reachability (Friends of Friends).
    4.  **Applications (6.7 - 6.9)**: Scheduling tasks (Partial Orders) and Clustering data (Equivalence).
    """)
    st.info("👈 Use the tabs above to navigate.")

# --- Tab 1: Basics (Bridge to DB) ---
def render_basics():
    st.subheader("1. The Bridge: Sets ↔ Tables ↔ Properties")
    
    # 交互区：定义关系
    with st.expander("🛠️ Interactive Lab: Define Relation", expanded=True):
        c1, c2, c3 = st.columns([1,1,2])
        A = parse_set_input(c1.text_input("Set A", "1, 2, 3, 4"))
        B = parse_set_input(c2.text_input("Set B", "1, 2, 3, 4"))
        rule = c3.selectbox("Rule", ["Divides (a | b)", "Less Than (a < b)", "Equal (a = b)"])
    
    if A and B:
        rel = generate_relation_data(A, B, rule)
        # 保存状态供其他模块使用
        st.session_state['rel_data'] = {'A': A, 'R': rel, 'rule': rule}
        
        c_math, c_mid, c_db = st.columns([4, 1, 5])
        with c_math:
            st.markdown("#### 📐 Math Notation")
            st.latex(f"R = {str(rel)}")
            
            # 属性检查
            props = check_properties(A, rel)
            st.caption("Properties (Section 6.2):")
            st.json(props)
            
        with c_db:
            st.markdown("#### 💾 Database Table")
            df = pd.DataFrame(rel, columns=["Attribute_A", "Attribute_B"])
            df.index += 1
            st.dataframe(df, use_container_width=True)
            st.markdown("""
            <div class='highlight-box'>
            Math <span class='math-tag'>Ordered Pair</span> = DB <span class='db-tag'>Tuple (Row)</span>
            </div>
            """, unsafe_allow_html=True)

# --- Tab 2: Modeling (Graphs & Matrices) ---
def render_modeling():
    st.subheader("2. Modeling: Visuals & Computation")
    st.markdown("Connecting **Section 6.3 (Digraphs)** & **Section 6.6 (Matrices)**.")
    
    if 'rel_data' not in st.session_state:
        st.warning("Please define a relation in the first tab to analyze its matrix.")
        # 使用默认数据演示
        nodes = [1, 2, 3]
        edges = [(1, 2), (2, 3)]
    else:
        nodes = st.session_state['rel_data']['A']
        edges = st.session_state['rel_data']['R']
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🕸️ Directed Graph (Digraph)")
        try:
            g = graphviz.Digraph(format='png')
            g.attr(rankdir='LR')
            for n in nodes: g.node(str(n))
            for u, v in edges: g.edge(str(u), str(v))
            st.graphviz_chart(g)
        except: st.error("Graphviz not installed.")
        
    with col2:
        st.markdown("#### 🔢 Adjacency Matrix (Computation)")
        matrix, idx_map = get_matrix(nodes, edges)
        df_mat = pd.DataFrame(matrix, columns=nodes, index=nodes)
        st.dataframe(df_mat.style.highlight_max(axis=None, color="#d1e7dd"), use_container_width=True)
        st.caption("1 means connected, 0 means not. This is how computers store graphs!")

    st.divider()
    
    # 结合 Section 6.5 (Reachability)
    st.markdown("#### 🚀 Power of Matrices (Section 6.5)")
    st.write("Finding 'Friends of Friends' (Paths of length k) using Matrix Multiplication.")
    
    k = st.slider("Path Length (k)", 1, 4, 2)
    m_pow = matrix_power(matrix, k)
    
    st.write(f"**Matrix Power $M^{k}$:** Shows reachability in exactly {k} steps.")
    st.dataframe(pd.DataFrame(m_pow, index=nodes, columns=nodes).style.highlight_max(axis=None, color='#ffecb3'))

# --- Tab 3: Operations (DB & Logic) ---
def render_operations():
    st.subheader("3. Operations: Logic & SQL")
    
    tab1, tab2 = st.tabs(["N-ary Relations (Databases)", "Composition (Logic)"])
    
    with tab1:
        st.markdown("**6.10 N-ary Relations (Meeting Focus)**")
        st.markdown("Murali Professor emphasized: **N-ary Relations are Tables**.")
        
        # Flight Data
        df = pd.DataFrame({
            "Flight": [101, 102, 201, 303],
            "Dep": ["Detroit", "Detroit", "Chicago", "New York"],
            "Arr": ["Chicago", "New York", "Detroit", "Miami"],
            "Time": ["08:00", "14:00", "09:30", "12:00"]
        })
        st.dataframe(df)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Selection (σ)**: Filter Rows")
            val = st.selectbox("Select Departure:", ["Detroit", "Chicago", "New York"])
            st.code(f"SELECT * FROM Flights WHERE Dep = '{val}'")
            st.dataframe(df[df["Dep"] == val])
        with c2:
            st.markdown("**Projection (π)**: Select Columns")
            cols = st.multiselect("Columns:", df.columns, ["Flight", "Dep"])
            if cols: 
                st.code(f"SELECT {', '.join(cols)} FROM Flights")
                st.dataframe(df[cols])
            else: st.warning("Pick a column")

    with tab2:
        st.markdown("**6.4 Composition ($R \circ S$)**")
        st.write("Used in Social Networks: 'Friend of a Friend'.")
        st.latex(r"x R y \land y S z \implies x (S \circ R) z")
        st.info("If user A follows B, and B follows C, then A is connected to C.")

# --- Tab 4: Applications (Scheduling & Clustering) ---
def render_applications():
    st.subheader("4. Advanced Applications")
    
    tab_sched, tab_clus = st.tabs(["Scheduler (Partial Order)", "Clustering (Equivalence)"])
    
    # 1. 调度器 (Topological Sort)
    with tab_sched:
        st.markdown("**6.7 & 6.8 Partial Orders & DAGs**")
        st.info("Problem: You have a list of courses with prerequisites. In what order should you take them?")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            tasks = {
                "CS1": [],
                "CS2": ["CS1"],
                "DataStruct": ["CS2"],
                "Algo": ["DataStruct"],
                "WebDev": ["CS1"]
            }
            st.json(tasks)
        
        with c2:
            try:
                g = graphviz.Digraph()
                edges = []
                all_nodes = list(tasks.keys())
                for course, prereqs in tasks.items():
                    g.node(course, style='filled', fillcolor='#fff3cd')
                    for p in prereqs:
                        g.edge(p, course)
                        edges.append((p, course))
                st.graphviz_chart(g)
            except: pass

        if st.button("🚀 Run Topological Sort"):
            sorted_plan = topological_sort(all_nodes, edges)
            if sorted_plan:
                st.success(f"Recommended Plan: {' → '.join(sorted_plan)}")
            else:
                st.error("Cycle Detected! Impossible to schedule.")

    # 2. 聚类器 (Hash/Equivalence)
    with tab_clus:
        st.markdown("**6.9 Equivalence Relations**")
        st.markdown("Grouping data into disjoint classes (Clustering/Hashing).")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            mod = st.number_input("Hash Function (Modulo N):", 2, 5, 3)
            num_range = st.slider("Data Range:", 1, 20, 10)
            numbers = list(range(1, num_range + 1))
        
        with col2:
            st.markdown(f"**Buckets:** Data $x$ goes to bucket $x \\pmod{{ {mod} }}$")
            clusters = {}
            for n in numbers:
                rem = n % mod
                if rem not in clusters: clusters[rem] = []
                clusters[rem].append(n)
            
            for rem, cluster in sorted(clusters.items()):
                st.info(f"**Bucket {rem}:** {cluster}")

# ==========================================
# 4. 主程序入口
# ==========================================
def main():
    st.title("Chapter 6: Relations")
    
    # 5个清晰的模块

    tabs = st.tabs([
        "Overview", 
        "1. Basics (The Bridge)", 
        "2. Modeling (Graph/Matrix)", 
        "3. Operations (Logic/DB)", 
        "4. Applications (Real-world)"
    ])

    with tabs[0]: render_overview()
    with tabs[1]: render_basics()
    with tabs[2]: render_modeling()
    with tabs[3]: render_operations()
    with tabs[4]: render_applications()

if __name__ == "__main__":
    main()
