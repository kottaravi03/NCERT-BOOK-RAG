import streamlit as st 

st.text_input("Name", key="name")
st.write(f"Hello {st.session_state.name}!")

def greet():
    st.write(f"call back sees: {st.session_state.name}!")

st.button("Greet", on_click=greet)