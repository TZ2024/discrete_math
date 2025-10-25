import streamlit as st
import pandas as pd
import re
import time
from itertools import product

st.set_page_config(page_title="Discrete Math App", layout="centered")

# Initialize persistent state for exercises if not already
if 'rel_solved' not in st.session_state:
    st.session_state['rel_solved'] = False
if 'tt_solved' not in st.session_state:
    st.session_state['tt_solved'] = False
if 'func_solved' not in st.session_state:
    st.session_state['func_solved'] = False

def fib_rec(n):
    """Recursive Fibonacci (naive)."""
    if n <= 1:
        return n
    return fib_rec(n-1) + fib_rec(n-2)

def fib_iter(n):
    """Iterative Fibonacci."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def page_relations():
    st.header("Relations")
    st.markdown("""
**Definition:** A **relation** between two sets A and B is any subset of the Cartesian product A×B.
In particular, a relation on a set A is a subset of A×A (pairs of elements from A).
For example, if A = {1,2,3}, one relation on A is the "less than" relation:
""")
    # Symbolic representation using LaTeX
    st.latex(r"R = \{(x,y) \in A \times A \mid x < y\}")
    st.write("This relation $R$ contains all ordered pairs $(x,y)$ where $x<y$ (with $x,y \in A$).")
    st.subheader("Representations of R")
    # Set of ordered pairs representation
    st.markdown("- **Set of ordered pairs:** " + r"$R = \{(1,2), (1,3), (2,3)\}$.")
    # Table (matrix) representation
    st.markdown("- **Table (matrix):** Rows and columns are elements of A; a **1** indicates that the pair is in R (true), and **0** indicates it is not.")
    A = [1, 2, 3]
    R = [(1, 2), (1, 3), (2, 3)]
    matrix = []
    for x in A:
        row = []
        for y in A:
            row.append(1 if (x, y) in R else 0)
        matrix.append(row)
    df = pd.DataFrame(matrix, index=A, columns=A)
    st.table(df)
    st.caption("Adjacency matrix for R (entry at row x, col y is 1 if xRy).")
    # Digraph representation
    st.markdown("- **Digraph:** We can visualize R as a directed graph, where each element in A is a node, and an arrow from x to y indicates $(x,y) \\in R$.")
    dot_str = "digraph G { 1; 2; 3; 1 -> 2; 1 -> 3; 2 -> 3; }"
    st.graphviz_chart(dot_str)
    st.caption("Directed graph of R (\"less than\" on A = {1,2,3}).")
    # Properties exercise
    st.subheader("Properties of this relation")
    st.write("Now, determine which properties R has:")
    st.write(" - **Reflexive:** Every element is related to itself (for all x in A, (x,x) ∈ R).")
    st.write(" - **Symmetric:** If (x,y) ∈ R, then (y,x) ∈ R.")
    st.write(" - **Antisymmetric:** If (x,y) ∈ R and (y,x) ∈ R, then x = y.")
    st.write(" - **Transitive:** If (x,y) ∈ R and (y,z) ∈ R, then (x,z) ∈ R.")
    if st.session_state['rel_solved']:
        st.success("✅ You have correctly identified the properties of R!")
    # Checkboxes for user input
    ref = st.checkbox("Reflexive", key="reflexive_cb")
    sym = st.checkbox("Symmetric", key="symmetric_cb")
    antisym = st.checkbox("Antisymmetric", key="antisymmetric_cb")
    trans = st.checkbox("Transitive", key="transitive_cb")
    # Check answer
    if st.button("Check Properties", key="check_rel"):
        if (not ref) and (not sym) and antisym and trans:
            st.success("Correct! R is **transitive** and **antisymmetric**, but **not reflexive** or **symmetric**.")
            st.session_state['rel_solved'] = True
        else:
            st.error("One or more properties are incorrect. Please try again.")

def page_logic():
    st.header("Truth Tables and Logic Expressions")
    st.write("In **propositional logic**, statements (propositions) can be either true or false. Logical operators like **and**, **or**, **not** allow us to build compound expressions and determine their truth values under different conditions.")
    st.write("A **truth table** lists all possible combinations of truth values for the input propositions and the resulting truth value of the compound expression for each combination.")
    # Input for expression and generate truth table
    expr_input = st.text_input("Enter a logical expression (use `and`, `or`, `not` with variables like p, q, r):", key="expr_input")
    if st.button("Generate Truth Table"):
        expr = expr_input.strip()
        if expr == "":
            st.error("Please enter a logical expression.")
        else:
            # Identify propositional variables
            expr_low = expr.lower()
            tokens = re.findall(r'\b[a-z]+\b', expr_low)
            variables = sorted({t for t in tokens if t not in {"and", "or", "not", "true", "false"}})
            if len(variables) == 0:
                st.error("No propositional variables found. Use letters like p, q, r as variables.")
            else:
                st.write(f"**Expression:** `{expr}`")
                st.write("**Truth Table:**")
                # Generate all combinations of truth values (True/False) for these variables
                combos = list(product([True, False], repeat=len(variables)))
                table_data = []
                for combo in combos:
                    env = {"True": True, "False": False}
                    # assign each variable a boolean value from combo
                    for var, val in zip(variables, combo):
                        env[var] = val
                    try:
                        result = eval(expr_low, {"__builtins__": None}, env)
                    except Exception as e:
                        st.error("Unable to evaluate expression. Please check your syntax.")
                        result = None
                    # Prepare row with truth values and result
                    row = [("True" if val else "False") for val in combo]
                    row.append("True" if result else "False")
                    table_data.append(row)
                col_names = [v.upper() for v in variables] + ["Result"]
                df = pd.DataFrame(table_data, columns=col_names)
                st.table(df)
    # Exercise: complete a truth table
    st.subheader("Exercise: Complete a Truth Table")
    if st.session_state['tt_solved']:
        st.success("✅ You have already completed this truth table correctly!")
    st.write("Fill in the truth values for the expression **not (p and q)** (the negation of p ∧ q):")
    # Display table structure with inputs
    col1, col2, col3 = st.columns([1, 1, 2])
    col1.markdown("**p**")
    col2.markdown("**q**")
    col3.markdown("**not (p and q)**")
    combos = [(True, True), (True, False), (False, True), (False, False)]
    answers = []
    for i, (p_val, q_val) in enumerate(combos):
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.write("True" if p_val else "False")
        c2.write("True" if q_val else "False")
        ans = c3.selectbox("", ["", "True", "False"], index=0, key=f"tt_ans_{i}", label_visibility="collapsed")
        answers.append(ans)
    if st.button("Check Answers", key="check_tt"):
        if "" in answers:
            st.error("Please select True/False for every row.")
        else:
            correct_results = [("True" if not (p and q) else "False") for p, q in combos]
            if answers == correct_results:
                st.success("Correct! `not (p and q)` is False only when both p and q are True, and True otherwise. (This matches **De Morgan's law**: it's equivalent to `not p or not q`.)")
                st.session_state['tt_solved'] = True
            else:
                st.error("One or more entries are incorrect. Try again.")

def page_functions():
    st.header("Functions")
    st.write("A **function** f is a rule that assigns each input exactly one output. The set of all possible inputs is the **domain** of f, and the set of possible outputs is the **range** of f.")
    st.write("For example, consider f(x) = x² on the integers. The domain is all integers, and the range is all non-negative integers (since squaring any integer is ≥ 0).")
    st.write("Below are some example functions defined in code. Determine the domain and range for each:")
    # Example functions in code
    st.markdown("**Function A:** (Double an integer)")
    st.code("def double(x):\n    return 2 * x", language="python")
    st.markdown("**Function B:** (Check if an integer is even)")
    st.code("def is_even(x):\n    return x % 2 == 0", language="python")
    st.markdown("**Function C:** (Length of a string)")
    st.code("def length(s):\n    return len(s)", language="python")
    st.markdown("**Options:**")
    st.markdown("1. Domain: All integers; Range: All integers\n2. Domain: All integers; Range: {False, True}\n3. Domain: All strings; Range: All non-negative integers")
    if st.session_state['func_solved']:
        st.success("✅ You already matched these correctly!")
    # Selection inputs for matching
    selA = st.selectbox("Function A's domain and range:", ["", "Option 1", "Option 2", "Option 3"], key="selA")
    selB = st.selectbox("Function B's domain and range:", ["", "Option 1", "Option 2", "Option 3"], key="selB")
    selC = st.selectbox("Function C's domain and range:", ["", "Option 1", "Option 2", "Option 3"], key="selC")
    if st.button("Check Answers", key="check_func"):
        if selA == "" or selB == "" or selC == "":
            st.error("Please select an option for all functions.")
        else:
            # Correct mapping: A-1, B-2, C-3
            if selA == "Option 1" and selB == "Option 2" and selC == "Option 3":
                st.success("Correct! A: Domain = all integers, Range = all integers; B: Domain = all integers, Range = {False, True}; C: Domain = all strings, Range = all non-negative integers.")
                st.session_state['func_solved'] = True
            else:
                st.error("One or more matches are incorrect. Try again.")

def page_growth():
    st.header("Growth of Functions: Fibonacci Example")
    st.write("Algorithms can differ greatly in efficiency as input size grows. The Fibonacci sequence is a classic example to compare a **recursive** vs an **iterative** approach.")
    st.write("Fibonacci definition: Fib(0)=0, Fib(1)=1, and Fib(n)=Fib(n-1)+Fib(n-2) for n ≥ 2.")
    st.write("The naive recursive implementation recomputes many subproblems, leading to **exponential** time complexity (~O(φ^n), φ≈1.618). The iterative implementation runs in **linear** time (O(n)).")
    st.write("Select a value of n and compare the computation time of Fibonacci(n) using recursion vs iteration:")
    n_val = st.slider("Choose n:", min_value=0, max_value=35, value=10)
    if st.button("Run Comparison"):
        start = time.time()
        fib_rec_result = fib_rec(n_val)
        t_rec = time.time() - start
        start = time.time()
        fib_iter_result = fib_iter(n_val)
        t_iter = time.time() - start
        # Display results and times
        st.write(f"Fib({n_val}) = {fib_iter_result}")
        st.write(f"Recursive method time: {t_rec:.6f} seconds")
        st.write(f"Iterative method time: {t_iter:.6f} seconds")
        speedup = (t_rec / t_iter) if t_iter > 0 else float('inf')
        if speedup > 1:
            st.write(f"Recursive vs Iterative time ratio ≈ {speedup:.1f}:1")
        st.write("As n increases, the recursive solution's runtime grows exponentially, while the iterative solution grows much more slowly.")

# Sidebar navigation
st.title("Discrete Mathematics Learning App")
page = st.sidebar.selectbox("Navigate to:", ["1. Relations", "2. Truth Tables and Logic Expressions", "3. Functions", "4.Growth of Functions"])
if page == "1. Relations":
    page_relations()
elif page == "2. Truth Tables and Logic Expressions":
    page_logic()
elif page == "3. Functions":
    page_functions()
elif page == "4.Growth of Functions":
    page_growth()
