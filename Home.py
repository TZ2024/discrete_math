import streamlit as st

st.set_page_config(
    page_title="Discrete Math Bridge",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Discrete Math Learning System")
st.markdown("### Bridging Mathematics & Computer Science")

st.info("👈 Please select a Chapter from the sidebar to begin.")

st.markdown("""
### Course Roadmap

#### 🟢 **Chapter 6: Relations (Available Now)**
We have transformed textbook concepts into **4 Interactive CS Modules**:
* **The Bridge**: Why `Relation` ≈ `SQL Table`.
* **Modeling**: Visualizing Social Networks with Digraphs & Matrices.
* **Operations**: How `Composition` explains "Friends of Friends".
* **Applications**: 
    * **Task Scheduling** (using Topological Sort on DAGs).
    * **Data Clustering** (using Equivalence Relations).

#### 🚧 **Chapter 1: Logic (Coming Soon)**
* Logic Gates & Circuits.
* Truth Tables as Data Validation.
""")