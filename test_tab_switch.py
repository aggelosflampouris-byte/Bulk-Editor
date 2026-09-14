import streamlit as st
import streamlit.components.v1 as components

tab1, tab2 = st.tabs(["Tab 1", "Video URL"])
with tab1:
    if st.button("Switch to Video URL"):
        st.session_state["redirect"] = True
        st.rerun()
    
    if st.session_state.pop("redirect", False):
        components.html("""
            <script>
                const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"] p');
                for (let i = 0; i < tabs.length; i++) {
                    if (tabs[i].textContent.includes("Video URL")) {
                        tabs[i].parentElement.parentElement.click();
                        break;
                    }
                }
            </script>
        """, height=0)

with tab2:
    st.write("Welcome to Video URL")
