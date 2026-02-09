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
    """解析输入并去重排序"""
    try: return sorted(list(set([int(x.strip()) for x in input_str.split(',') if x.strip()])))
    except: return [1, 2, 3, 4]

def format_set_display(s):
    """符号修正：将 [1, 2] 显示为 {1, 2}"""
    return "{" + ", ".join(map(str, s)) + "}"

def format_relation_display(rel):
    """符号修正：将 [(1,2)] 显示为 {(1,2)}"""
    if not rel: return "{}"
    items = ", ".join([f"({a},{b})" for a, b in rel])
    return f"\\{{ {items} \\}}"

def generate_relation_data(set_a, set_b, rule):
    relation = []
    for a in set_a:
        for b in set_b:
            is_related = False
            if rule == "Less Than (a < b)": is_related = (a < b)
            elif rule == "Greater Than (a > b)": is_related = (a > b)
            elif rule == "Equal (a = b)": is_related = (a == b)
            elif rule == "Divides (a | b)": is_related = (a != 0 and b % a == 0)
            elif rule == "Same Parity (a % 2 == b % 2)": is_related = (a % 2 == b % 2)
            # 【新功能】Murali 教授建议的例子：a = b - 1 (即 b 是 a 的直接后继)
            # 这对于展示 Transitive Closure 非常完美，因为 M^1 != M^2
            elif rule == "Immediate Predecessor (a = b - 1)": is_related = (a == b - 1)
            
            if is_related: relation.append((a, b))
    return relation

def check_properties(A, R_list):
    """计算属性：Reflexive, Symmetric, Anti-symmetric, Transitive"""
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
        if u in idx_map and v in idx_map:
            matrix[idx_map[u]][idx_map[v]] = 1
    return matrix, idx_map

def matrix_power(matrix, k):
    res = matrix
    for _ in range(k-1):
        res = np.dot(res, matrix)
    return (res > 0).astype(int) 



def transitive_closure(matrix):
    """Compute transitive closure M⁺ (reachability with path length >= 1) using Warshall's algorithm (boolean)."""
    reach = (matrix > 0).astype(int).copy()
    n = reach.shape[0]
    for k in range(n):
        # if i can reach k and k can reach j, then i can reach j
        reach = ((reach | (reach[:, [k]] & reach[[k], :])) > 0).astype(int)
    return reach

def topological_sort(nodes, edges):
    """带环检测的拓扑排序算法"""
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
        return None 
    return sorted_list

# ==========================================
# 3. 模块渲染函数
# ==========================================

def render_overview():
    st.header("Chapter 6: Relations as Computational Structures")
    st.markdown("""
    ### From Textbook to Interactive Tool
    Based on our meetings, this app transforms static math concepts into an **active CS playground** connecting 4 key areas:
    
    1.  **Sets ↔ SQL (The Bridge)**: 
        * *Concept*: Relations are just **Database Tables**.
        * *Sections*: 6.1, 6.2, 6.10.
    2.  **Modeling (Visuals)**: 
        * *Concept*: Representing connections ($V \\times V$) using **Digraphs** and **Adjacency Matrices**.
        * *Sections*: 6.3, 6.6.
    3.  **Operations (Logic)**: 
        * *Concept*: How `Composition` and Matrix Multiplication explain **"Friends of Friends"**.
        * *Sections*: 6.4, 6.5.
    4.  **Applications (Real-world)**: 
        * *Concept*: **Task Scheduling** (Topological Sort with Cycle Detection) and **Data Clustering**.
        * *Sections*: 6.7 - 6.9.
    """)
    st.info("👈 Select a module from the tabs above to start experimenting.")

# --- Tab 1: Basics ---
def render_basics():
    st.subheader("1. The Bridge: Sets ↔ Tables")
    st.markdown("Focus: **Ordered Pairs** notation & **SQL Table** representation.")
    
    with st.expander("🛠️ Define Relation (Set A & Set B)", expanded=True):
        c1, c2, c3 = st.columns([1,1,2])
        A = parse_set_input(c1.text_input("Set A Input (e.g. 1, 2, 3)", "1, 2, 3, 4"))
        B = parse_set_input(c2.text_input("Set B Input (e.g. 1, 2, 3)", "1, 2, 3, 4"))
        
        rule = c3.selectbox("Relation Rule", [
            "Divides (a | b)", 
            "Less Than (a < b)", 
            "Greater Than (a > b)",
            "Equal (a = b)", 
            "Same Parity (a % 2 == b % 2)",
            "Immediate Predecessor (a = b - 1)"
        ])
    
    if A and B:
        rel = generate_relation_data(A, B, rule)
        
        c_math, c_mid, c_db = st.columns([4, 1, 5])
        with c_math:
            st.markdown("#### 📐 Math Notation")
            st.latex(f"A = {format_set_display(A)}")
            st.latex(f"B = {format_set_display(B)}")
            st.markdown("**R (Ordered Pairs):**")
            st.latex(f"R = {format_relation_display(rel)}")
            st.caption("ℹ️ Properties (Reflexive/Symmetric) are now in the **Modeling** tab.")
            
        with c_db:
            st.markdown("#### 💾 Database Table")
            df = pd.DataFrame(rel, columns=["Attribute_A", "Attribute_B"])
            df.index += 1
            st.dataframe(df, use_container_width=True)
            st.markdown("""
            <div class='highlight-box'>
            Math <span class='math-tag'>Ordered Pair (a,b)</span> = DB <span class='db-tag'>Tuple (Row)</span>
            </div>
            """, unsafe_allow_html=True)

# --- Tab 2: Modeling ---
def render_modeling():
    st.subheader("2. Modeling: Properties, Graphs & Matrices")
    
    # 强制 V x V
    with st.expander("🕸️ Define Graph Nodes (Set V)", expanded=True):
        c1, c2 = st.columns([1, 2])
        nodes = parse_set_input(c1.text_input("Vertices V (e.g. 1, 2, 3, 4)", "1, 2, 3, 4"))
        
        # 【修改点】加入了新的规则，用于展示传递闭包
        rule = c2.selectbox("Edge Rule (on V × V)", [
            "Immediate Predecessor (a = b - 1)", # 教授推荐的例子！
            "Divides (a | b)", 
            "Less Than (a < b)", 
            "Greater Than (a > b)",
            "Equal (a = b)",
            "Same Parity (a % 2 == b % 2)"
        ])
        
        if rule == "Immediate Predecessor (a = b - 1)":
            st.info("💡 **Tip:** This relation is **NOT Transitive**. Try increasing the Path Length ($k$) below to see how connections grow ($M^1 \\neq M^2$)!")
    
    edges = generate_relation_data(nodes, nodes, rule)
    
    st.markdown("### 🔍 Analysis: Properties")
    props = check_properties(nodes, edges)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Reflexive", "Yes" if props['Reflexive'] else "No")
    p2.metric("Symmetric", "Yes" if props['Symmetric'] else "No")
    p3.metric("Anti-symmetric", "Yes" if props['Anti-symmetric'] else "No")
    p4.metric("Transitive", "Yes" if props['Transitive'] else "No")
    
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🕸️ Directed Graph")
        try:
            g = graphviz.Digraph(format='png')
            g.attr(rankdir='LR')
            for n in nodes: g.node(str(n))
            for u, v in edges: g.edge(str(u), str(v))
            st.graphviz_chart(g)
        except: st.error("Graphviz not installed.")
        
    with col2:
        st.markdown("#### 🔢 Adjacency Matrix")
        matrix, idx_map = get_matrix(nodes, edges)
        df_mat = pd.DataFrame(matrix, columns=nodes, index=nodes)
        st.dataframe(df_mat.style.highlight_max(axis=None, color="#d1e7dd"), use_container_width=True)

    st.markdown("#### 🚀 Reachability & Transitive Closure ($M^k$)")
    st.caption(
        "Tip: $M^k$ (boolean matrix product) shows whether there is a path of length exactly k. "
        "The transitive closure $M^+$ is $M \vee M^2 \vee \cdots \vee M^{n-1}$ (all reachable pairs). "
        "If a relation is **transitive**, then its closure equals itself (no new edges are added)."
    )
    n_nodes = len(nodes)
    k_max = max(1, min(8, n_nodes))  # keep UI friendly
    k = st.slider("Path Length (k)", 1, k_max, 1)

    m_k = matrix_power(matrix, k)
    m_plus = transitive_closure(matrix)

    tab_mk, tab_mplus = st.tabs([f"$M^{k}$ (length = {k})", "$M^{+}$ (Transitive Closure)"])

    with tab_mk:
        st.dataframe(
            pd.DataFrame(m_k, index=nodes, columns=nodes)
              .style.highlight_max(axis=None, color='#ffecb3'),
            use_container_width=True
        )
        if k == 1:
            st.write("This is the **adjacency matrix**. 1 means a direct edge.")
        else:
            st.write(f"1 means there is a path of length exactly **{k}** from row node to column node.")

    with tab_mplus:
        st.dataframe(
            pd.DataFrame(m_plus, index=nodes, columns=nodes)
              .style.highlight_max(axis=None, color='#d1e7dd'),
            use_container_width=True
        )

        # Explain what changed
        added = (m_plus == 1) & (matrix == 0)
        added_count = int(added.sum())
        if added_count == 0:
            st.success("✅ No new reachable pairs were added. This relation is already closed under reachability.")
        else:
            st.warning(f"➕ Transitive closure added {added_count} new reachable pairs (new edges you must add to make it transitive).")
            st.caption("These are exactly the pairs implied by chaining edges (paths) but missing as direct edges.")

    # small check for transitivity using matrices (R∘R ⊆ R)
    m2 = matrix_power(matrix, 2)
    if np.any((m2 == 1) & (matrix == 0)):
        st.info("Matrix check: $M^2$ has some 1s where $M$ has 0s, so the relation is **not transitive**.")
    else:
        st.info("Matrix check: $M^2 \subseteq M$, so the relation is **transitive**.")


# --- Tab 3: Operations ---
def render_operations():
    st.subheader("3. Operations: SQL & Logic")
    
    tab1, tab2 = st.tabs(["N-ary Relations (Databases)", "Composition (Logic)"])
    
    with tab1:
        st.markdown("**6.10 N-ary Relations**")
        df = pd.DataFrame({
            "Flight": [101, 102, 201, 303],
            "Dep": ["Detroit", "Detroit", "Chicago", "New York"],
            "Arr": ["Chicago", "New York", "Detroit", "Miami"],
            "Time": ["08:00", "14:00", "09:30", "12:00"]
        })
        # 强制从1开始计数
        df.index += 1
        st.dataframe(df)
        
        c1, c2 = st.columns(2)
        with c1:
            val = st.selectbox("Select Departure:", ["Detroit", "Chicago", "New York"])
            st.code(f"SELECT * FROM Flights WHERE Dep = '{val}'")
            filtered_df = df[df["Dep"] == val].copy()
            st.dataframe(filtered_df)
        with c2:
            cols = st.multiselect("Columns:", df.columns, ["Flight", "Dep"])
            if cols: 
                st.code(f"SELECT {', '.join(cols)} FROM Flights")
                st.dataframe(df[cols])

    with tab2:
        st.markdown("**6.4 Composition**")
        st.latex(r"x R y \land y S z \implies x (S \circ R) z")

# --- Tab 4: Applications ---
def render_applications():
    st.subheader("4. Advanced Applications")
    
    tab_sched, tab_clus = st.tabs(["Scheduler (Partial Order)", "Clustering (Equivalence)"])
    
    with tab_sched:
        st.markdown("**6.7 & 6.8 Partial Orders & DAGs**")
        st.info("Topological Sort: Finding a valid execution order for tasks.")

        # 【修改点】更新默认课程，增加 DiscreteMath，让 Algo 有2个先修课
        default_tasks = {
            "CS1": [], 
            "CS2": ["CS1"], 
            "DataStruct": ["CS2"], 
            "DiscreteMath": ["CS1"],         # 新增：离散数学 (依赖 CS1)
            "Algo": ["DataStruct", "DiscreteMath"], # 新增：Algo 现在有2个箭头指向它！
            "WebDev": ["CS1"]
        }
        
        # 鲁棒性测试
        st.markdown("#### 🧪 Test Robustness (Murali's Suggestion)")
        inject_cycle = st.checkbox("⚠️ Inject a Cycle (Make 'Algo' a prerequisite for 'CS1')")
        
        if inject_cycle:
            default_tasks["CS1"] = ["Algo"] # Cycle Created
            st.error("Cycle Injected! The graph is no longer a DAG.")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.json(default_tasks)
            if not inject_cycle:
                st.caption("Note: 'Algo' now has 2 prerequisites (DataStruct & DiscreteMath).")
        
        with c2:
            try:
                g = graphviz.Digraph()
                edges = []
                all_nodes = list(default_tasks.keys())
                for course, prereqs in default_tasks.items():
                    g.node(course, style='filled', fillcolor='#fff3cd')
                    for p in prereqs:
                        g.edge(p, course) # Arrow: Prereq -> Course
                        edges.append((p, course))
                st.graphviz_chart(g)
            except: pass

        if st.button("🚀 Run Topological Sort"):
            sorted_plan = topological_sort(all_nodes, edges)
            if sorted_plan:
                st.success(f"✅ Recommended Plan: {' → '.join(sorted_plan)}")
            else:
                st.error("⛔ Error: Cycle Detected! This is not a DAG. Scheduling is impossible.")

    with tab_clus:
        st.markdown("**6.9 Equivalence Relations**")
        st.success("Definition: Reflexive, **Symmetric**, and Transitive.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            mod = st.number_input("Hash Function (Modulo N):", 2, 5, 3)
            num_range = st.slider("Data Range:", 1, 20, 10)
            numbers = list(range(1, num_range + 1))
        
        with col2:
            st.markdown(f"**Buckets (Equivalence Classes):**")
            clusters = {}
            for n in numbers:
                rem = n % mod
                if rem not in clusters: clusters[rem] = []
                clusters[rem].append(n)
            
            for rem, cluster in sorted(clusters.items()):
                cluster_str = "{" + ", ".join(map(str, cluster)) + "}"
                st.info(f"**Class [{rem}]:** {cluster_str}")

# ==========================================
# 4. 主程序入口
# ==========================================
def main():
    st.title("Chapter 6: Relations")
    
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