import streamlit as st

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Tab 3"

st.segmented_control("Navigation", ["Tab 1", "Tab 2", "Tab 3"], key="active_tab", label_visibility="collapsed")

st.write(st.session_state["active_tab"])

if st.button("Go to Tab 2"):
    st.session_state["active_tab"] = "Tab 2"
    st.rerun()
