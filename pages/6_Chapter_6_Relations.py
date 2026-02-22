import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from collections import deque

# ==========================================
# 1. 页面配置与样式 (保持原有美观设计)
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
    return "{" + ", ".join(map(str, s)) + "}"

def format_relation_display(rel):
    if not rel: return "{}"
    items = ", ".join([f"({a},{b})" for a, b in rel])
    return f"\\{{ {items} \\}}"

def display_relation_smart(rel, label, prefix=None, max_latex=25):
    """
    智能显示关系：
    - 小于等于 max_latex：LaTeX 数学显示
    - 超过 max_latex：自动切换为表格，避免页面挤爆
    """
    st.markdown(f"#### {label}")

    if len(rel) == 0:
        if prefix: st.latex(prefix + r"\ \{\}")
        else: st.latex(r"\{\}")
        return

    if len(rel) <= max_latex:
        expr = format_relation_display(rel)
        if prefix: st.latex(prefix + r"\ " + expr)
        else: st.latex(expr)
        st.caption(f"Displayed as math notation ({len(rel)} pairs).")
    else:
        # Create a clean DataFrame for large relations
        df = pd.DataFrame(rel, columns=["a", "b"])
        df.index += 1
        st.dataframe(df, use_container_width=True)
        st.caption(f"Displayed as a table because there are {len(rel)} pairs.")

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
            elif rule == "Immediate Predecessor (a = b - 1)": is_related = (a == b - 1)
            
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
        if u in idx_map and v in idx_map:
            matrix[idx_map[u]][idx_map[v]] = 1
    return matrix, idx_map

def boolean_matmul(A, B):
    """Boolean matrix multiplication using OR over AND."""
    A_b = (A > 0)
    B_b = (B > 0)
    return ((A_b.astype(int) @ B_b.astype(int)) > 0).astype(int)


def matrix_power(matrix, k):
    base = (matrix > 0).astype(int)
    if k <= 1:
        return base
    res = base.copy()
    for _ in range(k - 1):
        res = boolean_matmul(res, base)
    return res


def compute_transitive_closure(matrix):
    n = len(matrix)
    base = (matrix > 0).astype(int)
    closure = base.copy()
    power_k = base.copy()
    for _ in range(1, n):
        power_k = boolean_matmul(power_k, base)
        closure = np.logical_or(closure, power_k).astype(int)
    return closure

def topological_sort(nodes, edges):
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
                if in_degree[dest] == 0: queue.append(dest)
    if len(sorted_list) != len(nodes): return None 
    return sorted_list

def compose_relations(R, S):
    R_set, S_set = set(R), set(S)
    result = set()
    R_out = {}
    for (a, b) in R_set: R_out.setdefault(a, set()).add(b)
    S_out = {}
    for (b, c) in S_set: S_out.setdefault(b, set()).add(c)
    for a, bs in R_out.items():
        for b in bs:
            for c in S_out.get(b, set()): result.add((a, c))
    return sorted(list(result))

def witness_middle_nodes(a, c, R, S):
    R_out = {}
    for (x, y) in R: R_out.setdefault(x, set()).add(y)
    S_in = {}
    for (y, z) in S: S_in.setdefault(z, set()).add(y)
    bs_from_R = R_out.get(a, set())
    bs_to_c_in_S = S_in.get(c, set())
    return sorted(list(bs_from_R.intersection(bs_to_c_in_S)))


def find_witness_path(nodes, edges, start, end):
    """Return one witness path start -> ... -> end if reachable, else []."""
    adj = {n: [] for n in nodes}
    for u, v in edges:
        if u in adj:
            adj[u].append(v)

    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nxt in adj.get(cur, []):
            if nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)

    if end not in prev:
        return []

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))

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
            "Divides (a | b)", "Less Than (a < b)", 
            "Greater Than (a > b)", "Equal (a = b)", 
            "Same Parity (a % 2 == b % 2)", "Immediate Predecessor (a = b - 1)"
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
        with c_db:
            st.markdown("#### 💾 Database Table")
            df = pd.DataFrame(rel, columns=["Attribute_A", "Attribute_B"]); df.index += 1
            st.dataframe(df, use_container_width=True)
            st.markdown("""<div class='highlight-box'>Math <span class='math-tag'>Ordered Pair (a,b)</span> = DB <span class='db-tag'>Tuple (Row)</span></div>""", unsafe_allow_html=True)

# --- Tab 2: Modeling (Smart Display Applied) ---
def render_modeling():
    st.subheader("2. Modeling: Properties, Graphs & Matrices")
    
    with st.expander("🕸️ Define Graph Nodes (Set V)", expanded=True):
        c1, c2 = st.columns([1, 2])
        nodes = parse_set_input(c1.text_input("Vertices V (e.g. 1, 2, 3, 4)", "1, 2, 3, 4"))
        rule = c2.selectbox("Edge Rule (on V × V)", [
            "Immediate Predecessor (a = b - 1)", "Divides (a | b)", 
            "Less Than (a < b)", "Greater Than (a > b)", 
            "Equal (a = b)", "Same Parity (a % 2 == b % 2)"
        ])
        if rule == "Immediate Predecessor (a = b - 1)":
            st.info("💡 **Tip:** This relation is **NOT Transitive**. Check the 'Transitive Closure Lab' tab to see edges grow!")
    
    edges = generate_relation_data(nodes, nodes, rule)
    matrix, _ = get_matrix(nodes, edges)
    props = check_properties(nodes, edges)

    tab_rep, tab_tc = st.tabs(["📊 Representations & Properties", "🧪 Transitive Closure Lab (M^k → M^+)"])

    with tab_rep:
        st.markdown("### 🔍 Analysis: Properties")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Reflexive", "Yes" if props['Reflexive'] else "No")
        p2.metric("Symmetric", "Yes" if props['Symmetric'] else "No")
        p3.metric("Anti-symmetric", "Yes" if props['Anti-symmetric'] else "No")
        p4.metric("Transitive", "Yes" if props['Transitive'] else "No")
        
        # --- SMART DISPLAY FOR BASE RELATION ---
        st.divider()
        display_relation_smart(
            edges, 
            "Relation R on V × V (Ordered Pairs)", 
            prefix=r"R =", 
            max_latex=25
        )

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🕸️ Directed Graph")
            try:
                g = graphviz.Digraph(format='png'); g.attr(rankdir='LR')
                for n in nodes: g.node(str(n))
                for u, v in edges: g.edge(str(u), str(v))
                st.graphviz_chart(g)
            except: st.error("Graphviz not installed.")
        with col2:
            st.markdown("#### 🔢 Adjacency Matrix")
            df_mat = pd.DataFrame(matrix, columns=nodes, index=nodes)
            st.dataframe(df_mat.style.highlight_max(axis=None, color="#d1e7dd"), use_container_width=True)
            # Pedagogical mapping note
            st.caption("Matrix entry M[i,j] = 1 exactly when (v_i, v_j) is in R.")

    with tab_tc:
        st.markdown("### 🧬 Transitive Closure Explorer")
        st.markdown("Use this lab to compare **exactly k-step reachability** (M^k) vs **overall reachability** (M^+).")

        example_mode = st.radio(
            "Choose example",
            ["Current Rule on V", "Predecessor Relation", "Flights Between Cities"],
            horizontal=True
        )

        base_nodes = nodes
        base_edges = edges

        if example_mode == "Predecessor Relation":
            n_pred = st.slider("Set size n (A={1..n})", 3, 12, min(8, max(3, len(nodes))))
            base_nodes = list(range(1, n_pred + 1))
            base_edges = [(a, b) for a in base_nodes for b in base_nodes if a == b - 1]
            st.caption("R = {(a,b) | a = b - 1}. Then R² captures distance-2 reachability.")

        elif example_mode == "Flights Between Cities":
            city_pool = ["Detroit", "Chicago", "NewYork", "Boston", "Seattle", "Austin", "Denver", "Miami"]
            city_count = st.slider("Number of cities", 5, 8, 6)
            base_nodes = city_pool[:city_count]
            default_flights = {
                "Detroit→Chicago": ("Detroit", "Chicago"),
                "Chicago→NewYork": ("Chicago", "NewYork"),
                "NewYork→Boston": ("NewYork", "Boston"),
                "Detroit→Austin": ("Detroit", "Austin"),
                "Austin→Denver": ("Austin", "Denver"),
                "Denver→Seattle": ("Denver", "Seattle"),
                "Miami→Boston": ("Miami", "Boston"),
            }
            chosen = st.multiselect(
                "Direct flights (relation R)",
                list(default_flights.keys()),
                default=[k for k in list(default_flights.keys())[:min(5, len(default_flights))]],
            )
            base_edges = [default_flights[k] for k in chosen if default_flights[k][0] in base_nodes and default_flights[k][1] in base_nodes]
            st.caption("R² means reachable in 2 flights (one layover). M^+ means reachable with any number of flights.")

        base_matrix, _ = get_matrix(base_nodes, base_edges)
        if len(base_nodes) < 2:
            st.info("Please provide at least 2 nodes.")
            return

        display_relation_smart(base_edges, "Base Relation R", prefix=r"R =", max_latex=25)

        show_steps = st.checkbox("Show steps (M¹, M², ..., up to n-1)", value=False)
        k = st.slider("Power k (show M^k)", 1, max(1, len(base_nodes) - 1), 1)

        if st.button("Compute Transitive Closure", key="compute_tc"):
            mk = matrix_power(base_matrix, k)
            m_plus = compute_transitive_closure(base_matrix)

            col_mk, col_plus = st.columns(2)
            with col_mk:
                st.markdown(f"#### M^{k} (exactly {k} steps)")
                st.dataframe(pd.DataFrame(mk, index=base_nodes, columns=base_nodes), use_container_width=True)
            with col_plus:
                st.markdown("#### M⁺ (transitive closure)")
                st.dataframe(pd.DataFrame(m_plus, index=base_nodes, columns=base_nodes), use_container_width=True)

            if show_steps:
                st.markdown("#### Step-by-step powers")
                cur = (base_matrix > 0).astype(int)
                for i in range(1, len(base_nodes)):
                    if i > 1:
                        cur = boolean_matmul(cur, (base_matrix > 0).astype(int))
                    st.markdown(f"M^{i}")
                    st.dataframe(pd.DataFrame(cur, index=base_nodes, columns=base_nodes), use_container_width=True)

            # interaction prompts
            st.markdown("### 🎯 Try-it Prompts")
            p1, p2, p3 = st.columns([1, 1, 2])
            s_node = p1.selectbox("Start", base_nodes, key="s_node_tc")
            e_node = p2.selectbox("End", base_nodes, index=min(1, len(base_nodes)-1), key="e_node_tc")
            guess = p3.radio("Predict reachability in M^+", ["Reachable", "Not reachable"], horizontal=True, key="guess_tc")

            idx_s, idx_e = base_nodes.index(s_node), base_nodes.index(e_node)
            reachable = (m_plus[idx_s][idx_e] == 1)
            if st.button("Check Prediction", key="check_pred_tc"):
                ok = (reachable and guess == "Reachable") or ((not reachable) and guess == "Not reachable")
                st.success("✅ Correct." if ok else "❌ Not this time.")
                if reachable:
                    path = find_witness_path(base_nodes, base_edges, s_node, e_node)
                    if path:
                        st.caption("Witness path: " + " → ".join(map(str, path)))

            # stabilization question
            stabilize_at = None
            prev = None
            cur = None
            for i in range(1, len(base_nodes) + 1):
                cur = matrix_power(base_matrix, i)
                if prev is not None and np.array_equal(cur, prev):
                    stabilize_at = i
                    break
                prev = cur.copy()

            if stabilize_at is None:
                stabilize_at = len(base_nodes)
            guess_step = st.slider("Guess when powers stabilize", 1, len(base_nodes), 2, key="guess_stable")
            if st.button("Check Stabilization", key="check_stable"):
                if guess_step == stabilize_at:
                    st.success(f"✅ Correct, stabilization starts at k={stabilize_at}.")
                else:
                    st.info(f"Close. For this graph, stabilization starts at k={stabilize_at}.")

            # add-one-edge impact
            st.markdown("#### Add one extra edge and compare closure")
            a_col, b_col = st.columns(2)
            add_u = a_col.selectbox("From", base_nodes, key="add_u")
            add_v = b_col.selectbox("To", base_nodes, key="add_v")
            if st.button("Apply extra edge", key="apply_extra"):
                new_edges = list(set(base_edges + [(add_u, add_v)]))
                new_m, _ = get_matrix(base_nodes, new_edges)
                new_plus = compute_transitive_closure(new_m)
                diff = ((new_plus == 1) & (m_plus == 0)).astype(int)
                st.dataframe(pd.DataFrame(diff, index=base_nodes, columns=base_nodes), use_container_width=True)
                st.caption("Cells with 1 are newly reachable pairs after adding the edge.")

# --- Tab 3: Operations (Smart Display Applied) ---
def render_operations():
    st.subheader("3. Operations: SQL & Logic")
    tab1, tab2 = st.tabs(["N-ary Relations (Databases)", "Composition (Logic)"])
    
    with tab1:
        st.markdown("**6.10 N-ary Relations**")
        df = pd.DataFrame({
            "Flight": [101, 102, 201, 303], "Dep": ["Detroit", "Detroit", "Chicago", "New York"],
            "Arr": ["Chicago", "New York", "Detroit", "Miami"], "Time": ["08:00", "14:00", "09:30", "12:00"]
        }); df.index += 1
        st.dataframe(df)
        c1, c2 = st.columns(2)
        with c1:
            val = st.selectbox("Select Departure:", ["Detroit", "Chicago", "New York"])
            st.code(f"SELECT * FROM Flights WHERE Dep = '{val}'")
            st.dataframe(df[df["Dep"] == val])
        with c2:
            cols = st.multiselect("Columns:", df.columns, ["Flight", "Dep"])
            if cols: st.code(f"SELECT {', '.join(cols)} FROM Flights"); st.dataframe(df[cols])

    with tab2:
        st.subheader("6.4 Composition: Friends of Friends")
        st.markdown("**Idea:** Composition creates new connections through an intermediate node.")
        st.latex(r"xRy \land ySz \Rightarrow x(S \circ R)z")
        st.latex(r"(x,z)\in(S\circ R)\iff \exists y\,(xRy \land ySz)")
        st.caption("Matrix connection: adjacency matrices satisfy  M(S∘R) = M(R) · M(S)  (booleanized).")

        with st.expander("🧩 Choose V, R, and S", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 1])
            V = parse_set_input(c1.text_input("Vertices V (e.g. 1,2,3,4)", "1, 2, 3, 4", key="comp_v"))
            rule_R = c2.selectbox("Rule for R", [
                "Immediate Predecessor (a = b - 1)", "Divides (a | b)", "Less Than (a < b)", 
                "Greater Than (a > b)", "Equal (a = b)", "Same Parity (a % 2 == b % 2)"
            ], key="comp_rule_R")
            rule_S = c3.selectbox("Rule for S", [
                "Immediate Predecessor (a = b - 1)", "Divides (a | b)", "Less Than (a < b)", 
                "Greater Than (a > b)", "Equal (a = b)", "Same Parity (a % 2 == b % 2)"
            ], key="comp_rule_S")

        if len(V) < 2:
            st.info("Please enter at least 2 vertices to see composition.")
        else:
            R = generate_relation_data(V, V, rule_R)
            S = generate_relation_data(V, V, rule_S)
            SoR = compose_relations(R, S)
            M_R, _ = get_matrix(V, R)
            M_S, _ = get_matrix(V, S)
            M_SoR_from_mats = boolean_matmul(M_R, M_S)
            M_SoR_rel, _ = get_matrix(V, SoR)

            st.divider()
            # --- SMART DISPLAY FOR COMPOSITION ---
            display_relation_smart(R, "Relation R (Ordered Pairs)", prefix=r"R =", max_latex=25)
            display_relation_smart(S, "Relation S (Ordered Pairs)", prefix=r"S =", max_latex=25)
            display_relation_smart(SoR, "Composition (S ∘ R) (Ordered Pairs)", prefix=r"S \circ R =", max_latex=25)

            st.divider()
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown("#### M(R)")
                st.dataframe(pd.DataFrame(M_R, index=V, columns=V), use_container_width=True)
            with col_m2:
                st.markdown("#### M(S)")
                st.dataframe(pd.DataFrame(M_S, index=V, columns=V), use_container_width=True)
            with col_m3:
                st.markdown("#### M(S ∘ R)")
                st.caption("Boolean Product: (M(R) · M(S)) > 0")
                st.dataframe(pd.DataFrame(M_SoR_from_mats, index=V, columns=V), use_container_width=True)

            if np.array_equal(M_SoR_rel, M_SoR_from_mats): st.success("✅ Match: Definition-based S∘R equals booleanized matrix product M(R)·M(S).")
            else: st.warning("⚠️ Mismatch detected. Check definitions.")

            st.divider()
            st.markdown("### 🔎 Explain the Middle Node y (Witness)")
            st.write("Pick **x** and **z**. We will show which **y** makes the composition true:")
            w1, w2 = st.columns(2)
            x_choice = w1.selectbox("Choose x (start)", V, key="witness_x")
            z_choice = w2.selectbox("Choose z (end)", V, key="witness_z")
            ys = witness_middle_nodes(x_choice, z_choice, R, S)
            if ys:
                st.success(f"✅ Yes. ({x_choice}, {z_choice}) is in S ∘ R.")
                st.write(f"**Middle node(s) y that make it work:** {', '.join(map(str, ys))}")
                support_rows = []
                for y in ys: support_rows.append({"(x,y) in R": f"({x_choice},{y})", "(y,z) in S": f"({y},{z_choice})"})
                st.dataframe(pd.DataFrame(support_rows), use_container_width=True)
            else:
                if (x_choice, z_choice) in set(SoR): st.warning("It seems reachable, but no witness y was found.")
                else: st.error(f"❌ No. ({x_choice}, {z_choice}) is NOT in S ∘ R.")

# --- Tab 4: Applications (Scheduler with Discrete Math) ---
def render_applications():
    st.subheader("4. Advanced Applications")
    tab_sched, tab_clus = st.tabs(["Scheduler (Partial Order)", "Clustering (Equivalence)"])
    
    with tab_sched:
        st.markdown("**6.7 & 6.8 Partial Orders & DAGs**")
        st.info("Topological Sort: Finding a valid execution order for tasks.")
        default_tasks = {
            "CS1": [], "CS2": ["CS1"], "DataStruct": ["CS2"], 
            "DiscreteMath": ["CS1"], "Algo": ["DataStruct", "DiscreteMath"], "WebDev": ["CS1"]
        }
        st.markdown("#### 🧪 Test Robustness")
        st.caption("Cycle mode is only for demonstrating the failure case. Keep OFF for valid scheduling.")
        inject_cycle = st.checkbox("⚠️ Inject a Cycle (Make 'Algo' a prerequisite for 'CS1')")
        if inject_cycle:
            default_tasks["CS1"] = ["Algo"]
            st.error("Cycle Injected! The graph is no longer a DAG.")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.json(default_tasks)
            if not inject_cycle: st.caption("Note: 'Algo' now has 2 prerequisites.")
        with c2:
            try:
                g = graphviz.Digraph(); g.attr(rankdir='LR')
                edges = []
                all_nodes = list(default_tasks.keys())
                for course, prereqs in default_tasks.items():
                    g.node(course, style='filled', fillcolor='#fff3cd')
                    for p in prereqs:
                        g.edge(p, course); edges.append((p, course))
                st.graphviz_chart(g)
            except: pass
        if st.button("🚀 Run Topological Sort"):
            sorted_plan = topological_sort(all_nodes, edges)
            if sorted_plan: st.success(f"✅ Recommended Plan: {' → '.join(sorted_plan)}")
            else: st.error("⛔ Error: Cycle Detected! This is not a DAG.")

        st.markdown("### 🧪 Micro Quiz")
        quiz_topo = st.radio("When does a topological ordering exist?", ["Only when graph is acyclic (DAG)", "For any directed graph"], key="quiz_topo")
        if st.button("Check Quiz", key="check_quiz_topo"):
            if quiz_topo == "Only when graph is acyclic (DAG)":
                st.success("Correct. Topological order exists iff the graph has no directed cycle.")
            else:
                st.error("Not correct. A directed cycle makes topological ordering impossible.")

    with tab_clus:
        st.markdown("**6.9 Equivalence Relations**")
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
                st.info(f"**Class [{rem}]:** " + "{" + ", ".join(map(str, cluster)) + "}")

# ==========================================
# 4. 主程序入口
# ==========================================
def main():
    st.title("Chapter 6: Relations")
    tabs = st.tabs(["Overview", "1. Basics (The Bridge)", "2. Modeling (Graph/Matrix)", "3. Operations (Logic/DB)", "4. Applications (Real-world)"])
    with tabs[0]: render_overview()
    with tabs[1]: render_basics()
    with tabs[2]: render_modeling()
    with tabs[3]: render_operations()
    with tabs[4]: render_applications()

if __name__ == "__main__":
    main()
