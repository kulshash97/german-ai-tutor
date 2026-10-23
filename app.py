import io
import os
import re
from google import genai
from google.genai import types
from gtts import gTTS
import streamlit as st

st.set_page_config(
    page_title="DeutschMeister | German AI Coach",
    page_icon="🇩🇪",
    layout="wide",
)

# API Authentication
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
  try:
    api_key = st.secrets.get("GEMINI_API_KEY")
  except Exception:
    api_key = None

with st.sidebar:
  st.title("🇩🇪 DeutschMeister")
  st.caption("First-Principles German Coach for Indian Learners")

  cefr_level = st.select_slider(
      "Target CEFR Level",
      options=[
          "A1 (Beginner)",
          "A2 (Elementary)",
          "B1 (Intermediate)",
          "B2 (Vantage)",
      ],
      value="A1 (Beginner)",
  )

  focus_mode = st.radio(
      "Study Track",
      [
          "Grammar & Sentence Structure",
          "Everyday Spoken Dialogue",
          "Vocabulary & Practical Phrases",
      ],
  )

  speech_speed = st.checkbox(
      "Slow Audio Pronunciation (Recommended for A1/A2)", value=True
  )

  if not api_key:
    st.divider()
    api_key = st.text_input("Enter Gemini API Key", type="password")

  if st.button("Clear Conversation", use_container_width=True):
    st.session_state.messages = []
    st.session_state.last_processed_audio = None
    st.rerun()

if not api_key:
  st.info(
      "👋 Willkommen! Please provide your **GEMINI_API_KEY** in the sidebar or"
      " via Streamlit Cloud Secrets to begin."
  )
  st.stop()

client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = f"""
You are 'Herr Lukas', an elite German tutor coaching an Indian student.
Target Level: {cefr_level}
Current Track: {focus_mode}

Core Pedagogy Rules:
1. Explain German concepts in clean English. Relate grammar to Indian languages (e.g., 'Sie' vs 'du' is like 'Aap' vs 'Tum'; German cases operate like Vibhakti / case-markers).
2. For German expressions, format clearly:
   - **German**: [German sentence here]
   - **Pronunciation**: [Simple English phonetic guide, e.g., Ikh hy-suh...]
   - **Meaning**: [English translation]
3. Keep it interactive and encouraging. Always conclude with 1 short practice question.
"""

if "messages" not in st.session_state:
  st.session_state.messages = [{
      "role": "assistant",
      "content": (
          f"**Guten Tag!** I am Herr Lukas. We will master German starting at"
          f" **{cefr_level}**.\n\nType your question, or use the microphone below"
          " to practice speaking!"
      ),
      "audio_bytes": None,
  }]

# Render History
for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])
    if msg.get("audio_bytes"):
      st.audio(msg["audio_bytes"], format="audio/mp3")

# Microphone Voice Input
audio_val = st.audio_input("🎙️ Practice Speaking (Tap to record):")

user_prompt = None

if audio_val is not None:
  audio_bytes = audio_val.read()
  if (
      "last_processed_audio" not in st.session_state
      or st.session_state.last_processed_audio != audio_bytes
  ):
    st.session_state.last_processed_audio = audio_bytes
    with st.spinner("Transcribing your speech..."):
      try:
        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav",
        )
        stt_res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                audio_part,
                types.Part.from_text(
                    text=(
                        "Transcribe ONLY the user's spoken words verbatim."
                        " Return plain text with no commentary or labels."
                    )
                ),
            ],
        )
        raw_stt = stt_res.text.strip()
        cleaned_stt = re.split(
            r"(?i)clean transcription:|the user asked|transcription:", raw_stt
        )[0].strip()
        user_prompt = cleaned_stt if cleaned_stt else raw_stt
      except Exception as err:
        st.error(f"Speech transcription error: {err}")

# Text Input
chat_prompt = st.chat_input("Type in English or German...")
if chat_prompt:
  user_prompt = chat_prompt

# Process and Respond
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
    with st.spinner("Herr Lukas is preparing the lesson..."):
      # Call 1: Generate Full Lesson
      res = client.models.generate_content(
          model="gemini-2.5-flash",
          contents=history_contents,
          config=types.GenerateContentConfig(
              system_instruction=SYSTEM_INSTRUCTION,
              temperature=0.2,
          ),
      )
      lesson_text = res.text
      st.markdown(lesson_text)

      # Call 2: Dedicated German Audio Phrase Extractor
      # This extracts ONLY the German words to pronounce, discarding all English explanations
      audio_bot_bytes = None
      try:
        extraction_res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_text(
                    text=f"""From the lesson text below, extract ONLY the primary German sentence or phrase that the student must practice speaking.
Output ONLY the clean German sentence. Do NOT include English words, do NOT include pronunciation brackets, do NOT include asterisks (*), and do NOT include labels.

Lesson:
{lesson_text}"""
                )
            ],
        )
        extracted_german = extraction_res.text.strip()

        # Hard Sanitization: Strip ANY remaining markdown asterisks, quotes, brackets, or labels
        clean_german = re.sub(
            r"[\*\_#\[\]\(\)<>\"'`]", "", extracted_german
        ).strip()
        clean_german = re.sub(
            r"(?i)^(german|deutsch|phrase|sentence):\s*", "", clean_german
        ).strip()
        # Ensure only the first line is taken
        clean_german = clean_german.split("\n")[0].strip()

        # Generate Audio only for the isolated phrase (2-3 seconds long)
        if clean_german and len(clean_german) > 2:
          tts = gTTS(text=clean_german, lang="de", slow=speech_speed)
          fp = io.BytesIO()
          tts.write_to_fp(fp)
          fp.seek(0)
          audio_bot_bytes = fp.read()
          st.audio(audio_bot_bytes, format="audio/mp3")
      except Exception:
        pass

      st.session_state.messages.append({
          "role": "assistant",
          "content": lesson_text,
          "audio_bytes": audio_bot_bytes,
      })