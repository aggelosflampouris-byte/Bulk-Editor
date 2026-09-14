import streamlit as st
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Tab 3"

tab = st.radio("Tabs", ["Tab 1", "Tab 2", "Tab 3"], horizontal=True, label_visibility="collapsed", key="active_tab")

if tab == "Tab 1":
    st.write("Tab 1")
elif tab == "Tab 2":
    st.write("Tab 2")
elif tab == "Tab 3":
    if st.button("Go to Tab 2"):
        st.session_state["active_tab"] = "Tab 2"
        st.rerun()
