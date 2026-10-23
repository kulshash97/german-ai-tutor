import io
import os
import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS

st.set_page_config(
    page_title="DeutschMeister | German AI Coach",
    page_icon="🇩🇪",
    layout="wide",
)

# --- API Authentication ---
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

# Sidebar Controls
with st.sidebar:
    st.title("🇩🇪 DeutschMeister")
    st.caption("First-Principles German Language Engine")

    cefr_level = st.select_slider(
        "Target CEFR Level",
        options=["A1 (Beginner)", "A2 (Elementary)", "B1 (Intermediate)", "B2 (Vantage)"],
        value="A1 (Beginner)",
    )

    focus_mode = st.radio(
        "Study Track",
        ["Grammar & Syntax Rules", "Spoken Dialogue Simulation", "Vocabulary & Idiomatic Phrases"],
    )

    if not api_key:
        st.divider()
        api_key = st.text_input("Enter Gemini API Key", type="password")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.info("👋 Willkommen! Please add your **GEMINI_API_KEY** in the sidebar or via Streamlit Cloud Secrets.")
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = f"""
You are 'Herr Lukas', an elite native German tutor teaching an English-speaking student.
Target Level: {cefr_level}
Current Track: {focus_mode}

Pedagogy Rules:
1. First-Principles: Explain grammar mechanically (e.g., cases, sentence brackets, verb-second position).
2. Format German output clearly:
   - German sentence (with gender: der/die/das).
   - Phonetic pronunciation guide in brackets (e.g., [Ikh hah-buh]).
   - Literal breakdown + idiomatic English translation.
3. Corrections: When the student makes an error, diagnose the rule first, show the fix in bold, and prompt them to re-try.
4. Keep dialogue turns short and interactive for audio playback.
"""

# Initialize History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": f"**Guten Tag!** I am Herr Lukas. We will master German starting at **{cefr_level}**.\n\nType a phrase, ask any grammar question, or record your voice below to practice speaking.",
        }
    ]

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio_bytes"):
            st.audio(msg["audio_bytes"], format="audio/mp3")

# Input controls: Native Streamlit Audio Input + Chat Input
audio_val = st.audio_input("🎙️ Practice Speaking (Record your voice):")

user_prompt = None

# Case 1: Spoken Audio Recorded
if audio_val is not None:
    # Process each recording once
    audio_bytes = audio_val.read()
    if "last_processed_audio" not in st.session_state or st.session_state.last_processed_audio != audio_bytes:
        st.session_state.last_processed_audio = audio_bytes
        with st.spinner("Transcribing your pronunciation..."):
            try:
                audio_part = types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav",
                )
                stt_res = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        audio_part,
                        "Transcribe the spoken audio verbatim in English or German. Return only the transcription without markdown or comments.",
                    ],
                )
                user_prompt = stt_res.text.strip()
            except Exception as err:
                st.error(f"Speech transcription error: {err}")

# Case 2: Standard Text Input
chat_prompt = st.chat_input("Type in English or German...")
if chat_prompt:
    user_prompt = chat_prompt

# Process message and generate bot reply
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    history_contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in st.session_state.messages[:-1]
    ]
    history_contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
    )

    with st.chat_message("assistant"):
        with st.spinner("Herr Lukas is formulating feedback..."):
            res = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=history_contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.3,
                ),
            )
            output_text = res.text
            st.markdown(output_text)

            # Generate pronunciation audio
            bot_audio_bytes = None
            try:
                tts = gTTS(text=output_text[:300], lang="de", slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                bot_audio_bytes = fp.read()
                st.audio(bot_audio_bytes, format="audio/mp3")
            except Exception:
                pass

            st.session_state.messages.append({
                "role": "assistant",
                "content": output_text,
                "audio_bytes": bot_audio_bytes,
            })