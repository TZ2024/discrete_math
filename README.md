# Discrete Math Interactive App (Chapter 6)

Interactive Streamlit app for teaching **Relations** with graphs, matrices, composition, and real-world applications.

## Scope (Chapter 6)

Implemented modules:
- **Basics (Sets ↔ Tables)**: ordered pairs and relation-as-table mapping.
- **Modeling (Graphs & Matrices)**:
  - Relation properties (reflexive, symmetric, anti-symmetric, transitive)
  - Adjacency matrix and digraph view
  - **Transitive Closure Lab** with `M^k`, `M^+`, and new-edge visualization
  - Prediction + reveal interaction and witness path display
- **Operations (Composition)**:
  - Definition-based composition `S ∘ R`
  - Boolean matrix product verification
  - Witness middle-node explanation
- **Advanced Applications**:
  - Course scheduling via topological sort
  - Cycle injection as optional failure-case demo
  - Equivalence classes via modulo clustering
  - Micro quiz blocks for engagement

## Instructor Feedback Implemented

- Added/kept explicit **Transitive Closure Lab** under Modeling with non-transitive growth behavior (`M^1, M^2, ... -> M^+`).
- Updated Advanced Applications graph so **Algorithms has two prerequisites**:
  - `DataStruct -> Algo`
  - `DiscreteMath -> Algo`
- Added clear cycle handling guidance and warning for topological sort.

## Run Locally

```bash
cd Desktop/GSRA/discrete_math
python3 -m pip install --user -r requirements.txt
python3 -m streamlit run Home.py
```

If `streamlit` is not on PATH, use:

```bash
python3 -m streamlit run Home.py
```

## Deploy (Streamlit Cloud)

- Repo should include `Home.py`, `pages/`, and `requirements.txt`.
- Set app entrypoint to `Home.py`.
- Redeploy after pushing changes.
