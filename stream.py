import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"  

st.set_page_config(page_title="Medical RAG Chatbot", layout="wide")

# ---------------------------
# SESSION STATE
# ---------------------------
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "chatroom_id" not in st.session_state:
    st.session_state.chatroom_id = None


# ---------------------------
# AUTH FUNCTIONS
# ---------------------------
def signup(email, password):
    return requests.post(f"{BACKEND_URL}/register", json={"email": email, "password": password})

def login(email, password):
    res = requests.post(f"{BACKEND_URL}/login", json={"email": email, "password": password})
    if res.ok and "user" in res.json():
        st.session_state.user_email = email
    return res


# ---------------------------
# SIDEBAR - AUTH
# ---------------------------
st.sidebar.title("Authentication")

if not st.session_state.user_email:
    choice = st.sidebar.radio("Choose action", ["Login", "Signup"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if choice == "Signup":
        if st.sidebar.button("Sign Up"):
            if email and password:
                res = signup(email, password)
                if res.ok:
                    st.sidebar.success("Signup successful! Please login.")
                else:
                    st.sidebar.error(f"Signup failed: {res.text}")
            else:
                st.sidebar.error("Please enter both email and password")
    else:  # login
        if st.sidebar.button("Login"):
            if email and password:
                res = login(email, password)
                if res.ok and "user" in res.json():
                    st.sidebar.success("Login successful!")
                    
                    # Create a new chatroom after login
                    create_res = requests.post(f"{BACKEND_URL}/new_chatroom/{email}")
                    if create_res.ok:
                        new_chatroom = create_res.json()
                        st.session_state.chatroom_id = new_chatroom["chatroom_id"]
                        st.sidebar.success(f"New chatroom #{new_chatroom['chatroom_id']} created!")
                        st.rerun()
                    else:
                        st.sidebar.error("Failed to create chatroom")
                else:
                    response_data = res.json() if res.ok else {"error": "Login failed"}
                    st.sidebar.error(response_data.get("error", "Invalid credentials"))
            else:
                st.sidebar.error("Please enter both email and password")
else:
    st.sidebar.success(f"Logged in as {st.session_state.user_email}")
    if st.sidebar.button("Logout"):
        st.session_state.user_email = None
        st.session_state.chatroom_id = None
        st.rerun()


# ---------------------------
# CHATROOMS LIST
# ---------------------------
if st.session_state.user_email:
    st.sidebar.subheader("Your Chatrooms")

    # Fixed: Use correct endpoint with email parameter
    res = requests.get(f"{BACKEND_URL}/chatrooms/{st.session_state.user_email}")
    if res.ok:
        chatrooms_data = res.json()
        chatrooms = chatrooms_data.get("chatrooms", [])
        if chatrooms:
            chatroom_names = [f"Chatroom {c[0]}" for c in chatrooms]  # c[0] is the id
            chatroom_map = {f"Chatroom {c[0]}": c[0] for c in chatrooms}
            
            # Set default selection if current chatroom_id exists
            default_index = 0
            if st.session_state.chatroom_id:
                current_name = f"Chatroom {st.session_state.chatroom_id}"
                if current_name in chatroom_names:
                    default_index = chatroom_names.index(current_name)
            
            selected = st.sidebar.selectbox("Select a chatroom", chatroom_names, index=default_index)
            st.session_state.chatroom_id = chatroom_map[selected]
        else:
            st.sidebar.info("No chatrooms yet.")
    else:
        st.sidebar.error("Failed to fetch chatrooms")

    if st.sidebar.button("Create New Chatroom"):
        res = requests.post(f"{BACKEND_URL}/new_chatroom/{st.session_state.user_email}")
        if res.ok:
            new_chatroom = res.json()
            st.session_state.chatroom_id = new_chatroom["chatroom_id"]
            st.sidebar.success(f"New chatroom #{new_chatroom['chatroom_id']} created!")
            st.rerun()


# ---------------------------
# DOCUMENT UPLOAD
# ---------------------------
if st.session_state.user_email:
    st.sidebar.subheader("📄 Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload medical documents", 
        type=["pdf", "docx", "csv"], 
        accept_multiple_files=True
    )
    
    if uploaded_files and st.sidebar.button("Process Documents"):
        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        try:
            res = requests.post(f"{BACKEND_URL}/upload", files=files)
            if res.ok:
                result = res.json()
                st.sidebar.success(f"✅ {result['message']}")
                for doc in result["documents"]:
                    if doc["status"] == "success":
                        st.sidebar.success(f"📄 {doc['filename']}: {doc['chunks_count']} chunks")
                    else:
                        st.sidebar.error(f"❌ {doc['filename']}: {doc['status']}")
            else:
                st.sidebar.error(f"Upload failed: {res.text}")
        except Exception as e:
            st.sidebar.error(f"Upload error: {str(e)}")


# ---------------------------
# MAIN CHAT INTERFACE
# ---------------------------
# ---------------------------
# MAIN CHAT INTERFACE
# ---------------------------
st.title("💊 Medical RAG Chatbot")

if st.session_state.user_email and st.session_state.chatroom_id:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"💬 Chat - Chatroom #{st.session_state.chatroom_id}")
        
        # Fetch chat history
        res = requests.get(f"{BACKEND_URL}/messages/{st.session_state.chatroom_id}")
        if res.ok:
            messages_data = res.json()
            messages = messages_data.get("messages", [])
            for msg in messages:
                user_message, ai_response = msg
                if user_message:
                    st.chat_message("user").markdown(user_message)
                if ai_response:
                    st.chat_message("assistant").markdown(ai_response)

        # ---- Sticky chat input ----
        with st.container():
            st.markdown(
                """
                <style>
                div[data-testid="stChatInput"] {
                    position: fixed;
                    bottom: 1rem;
                    left: 20rem; /* adjust if sidebar overlaps */
                    right: 1rem;
                    z-index: 100;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            if prompt := st.chat_input("Ask your medical question..."):
                st.chat_message("user").markdown(prompt)

                chat_data = {
                    "email": st.session_state.user_email,
                    "chatroom_id": st.session_state.chatroom_id,
                    "message": prompt
                }

                try:
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        with requests.post(f"{BACKEND_URL}/chat", json=chat_data, stream=True) as res:
                            if res.ok:
                                for chunk in res.iter_content(chunk_size=1, decode_unicode=True):
                                    if chunk:
                                        full_response += chunk
                                        response_placeholder.markdown(full_response + "▌")
                                response_placeholder.markdown(full_response)
                            else:
                                st.error(f"Chat failed: {res.text}")
                except Exception as e:
                    st.error(f"Chat error: {str(e)}")

    
    
else:
    st.info("👈 Please login to start using the medical chatbot.")
    