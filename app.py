import io
import os
import re
from google import genai
from google.genai import types
from google.genai.errors import APIError
from gtts import gTTS
import streamlit as st

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
1. Explain German concepts clearly in English. Relate grammar to Indian languages (e.g., 'Sie' vs 'du' is like 'Aap' vs 'Tum'; German cases operate like Vibhakti / case-markers).
2. For German phrases, format clearly in markdown:
   - **German**: [German sentence]
   - **Pronunciation**: [Simple English phonetic guide]
   - **Meaning**: [English translation]
3. STRICT AUDIO OUTPUT RULE:
   At the absolute end of your response, output a single delimiter block containing ONLY the German phrase that should be spoken aloud:
   <<<GERMAN_ONLY>>>
   [Only 1 pure German sentence. No English, no asterisks, no quotes, no brackets]
   <<<END_GERMAN>>>
"""

if "messages" not in st.session_state:
  st.session_state.messages = [{
      "role": "assistant",
      "content": (
          f"**Guten Tag!** I am Herr Lukas. We will master German starting at"
          f" **{cefr_level}**.\n\nType your question, or use the microphone"
          " below to practice speaking!"
      ),
      "audio_bytes": None,
  }]

# Display message history
for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])
    if msg.get("audio_bytes"):
      st.audio(msg["audio_bytes"], format="audio/mp3")

# Microphone voice input
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

# Text input
chat_prompt = st.chat_input("Type in English or German...")
if chat_prompt:
  user_prompt = chat_prompt

# Helper to build strictly valid alternating history
def build_clean_history(messages, current_prompt):
  valid_contents = []
  last_role = None

  for m in messages:
    role = "user" if m["role"] == "user" else "model"
    text = str(m.get("content", "")).strip()
    if not text:
      continue
    # Prevent consecutive identical roles which triggers 400 ClientError
    if role == last_role:
      continue
    valid_contents.append(
        types.Content(role=role, parts=[types.Part.from_text(text=text)])
    )
    last_role = role

  # Ensure the final message is the current user prompt
  if last_role == "user" and valid_contents:
    valid_contents.pop()

  valid_contents.append(
      types.Content(
          role="user", parts=[types.Part.from_text(text=str(current_prompt))]
      )
  )
  return valid_contents


# Process and Respond
if user_prompt:
  st.session_state.messages.append({"role": "user", "content": user_prompt})
  with st.chat_message("user"):
    st.markdown(user_prompt)

  clean_history = build_clean_history(st.session_state.messages[:-1], user_prompt)

  with st.chat_message("assistant"):
    with st.spinner("Herr Lukas is preparing the lesson..."):
      try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=clean_history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )
        raw_output = res.text
      except APIError as e:
        st.error(
            "Service temporarily busy. Please re-send your message in a few"
            " moments."
        )
        st.stop()

      # Extract pure German phrase from delimiter
      audio_phrase = ""
      match = re.search(
          r"<<<GERMAN_ONLY>>>(.*?)<<<END_GERMAN>>>", raw_output, re.DOTALL
      )
      if match:
        audio_phrase = match.group(1).strip()
        display_text = re.sub(
            r"<<<GERMAN_ONLY>>>.*?<<<END_GERMAN>>>",
            "",
            raw_output,
            flags=re.DOTALL,
        ).strip()
      else:
        display_text = raw_output
        # Fallback: grab first bold phrase if delimiter was skipped
        bold_matches = re.findall(r"\*\*([A-ZÄÖÜa-zäöüß\s]+)\*\*", raw_output)
        if bold_matches:
          audio_phrase = bold_matches[0]

      # STRICT SANITIZATION: Completely strip asterisks, markdown, brackets, and labels
      clean_audio = re.sub(r"[\*\_#\[\]\(\)<>\"'`]", "", audio_phrase).strip()
      clean_audio = re.sub(
          r"(?i)^(german|deutsch|phrase|sentence):\s*", "", clean_audio
      ).strip()
      clean_audio = clean_audio.split("\n")[0].strip()

      st.markdown(display_text)

      # Generate Audio: Only 2-3 seconds long for the pure German sentence
      bot_audio_bytes = None
      if clean_audio and len(clean_audio) > 2:
        try:
          tts = gTTS(text=clean_audio, lang="de", slow=speech_speed)
          fp = io.BytesIO()
          tts.write_to_fp(fp)
          fp.seek(0)
          bot_audio_bytes = fp.read()
          st.audio(bot_audio_bytes, format="audio/mp3")
        except Exception:
          pass

      st.session_state.messages.append({
          "role": "assistant",
          "content": display_text,
          "audio_bytes": bot_audio_bytes,
      })