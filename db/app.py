import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Simple Multipage App",
    page_icon="📚",
    layout="wide"
)


# Main page content
def main():
    st.title("Welcome to the Main Page")
    st.write("This is the main page of our simple multipage Streamlit application.")

    st.markdown("""
    ### Navigation
    Use the sidebar to navigate between pages:
    - **Main Page** (You are here)
    - **Page 1** - Additional content
    - **Page 2** - More information
    """)

    st.markdown("---")
    st.write("This is a simple demonstration of Streamlit's multipage capabilities.")


# Create the app structure
if __name__ == "__main__":
    main()