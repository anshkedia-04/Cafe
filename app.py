import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA


# === Utility Functions ===
@st.cache_resource
def load_docs():
    pdf_loader = PyPDFLoader("menu.pdf")
    txt_loader = TextLoader("faq.txt")
    csv_loader = CSVLoader("offers.csv")
    return pdf_loader.load() + txt_loader.load() + csv_loader.load()

@st.cache_resource
def create_vectorstore(_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs_split = splitter.split_documents(_docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs_split, embeddings)

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    return HuggingFacePipeline(pipeline=pipe)


# === Main App ===
def main():
    st.set_page_config(page_title="BrewBot Café Assistant", layout="wide")
    
    # Layout: Menu/Info on left, Chatbot on right
    left, right = st.columns([1, 2])

    # Left Column: Static Menu/Info
    with left:
        st.image("cafe_logo.png", use_column_width=True)
        st.markdown("### 📋 Full Menu (From `menu.pdf`)")
        try:
            with open("menu.pdf", "rb") as f:
                base64_pdf = f.read()
                st.download_button("📥 Download Menu", "menu.pdf", file_name="menu.pdf")

            # Optional: show plain text version
            docs = load_docs()
            for d in docs:
                if "menu" in d.metadata['source'].lower():
                    st.markdown(d.page_content[:1000])  # show a preview
                    break
        except:
            st.error("❌ Could not display the menu.")

        st.markdown("---")
        st.markdown("## 🌐 Connect with us")
        st.markdown("""
<a href="https://instagram.com" target="_blank">
    <img src="https://img.icons8.com/ios-filled/25/000000/instagram-new.png" style="margin-right: 10px;">
</a>
<a href="https://twitter.com" target="_blank">
    <img src="https://img.icons8.com/ios-filled/25/000000/twitter.png" style="margin-right: 10px;">
</a>
<a href="https://github.com/anshkedia-04/BrewBot" target="_blank">
    <img src="https://img.icons8.com/ios-filled/25/000000/github.png" style="margin-right: 10px;">
</a>
<a href="https://streamlit.io" target="_blank">
    <img src="https://img.icons8.com/ios-filled/25/000000/streamlit.png" style="margin-right: 10px;">
</a>
<a href="https://www.langchain.com" target="_blank">
    <img src="https://img.icons8.com/ios-filled/25/000000/code.png">
</a>
""", unsafe_allow_html=True)

        st.markdown("📧 [support@brewbot.com](mailto:support@brewbot.com)")
        st.markdown("🧪 Version: `1.0.0`")
        st.markdown("❤️ Made with love by **BrewBot Team**")
        st.markdown("© 2025 BrewBot Inc. All rights reserved.")

    # Right Column: Chatbot
    with right:
        st.title("☕ BrewBot : Café Assistant Chatbot")
        st.markdown("Ask me anything about the menu, FAQs, or current offers!")

        with st.spinner("📄 Loading documents and setting up the model..."):
            docs = load_docs()
            db = create_vectorstore(docs)
            llm = load_llm()
            retriever = db.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            st.success("✅ Documents loaded successfully!")

        query = st.text_input("🔎 Enter your question:")
        if query:
            with st.spinner("Generating answer..."):
                try:
                    result = qa_chain.invoke(query)
                    st.success("💬 Answer: " + result["result"])
                except Exception as e:
                    st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
