"""
Ask JIVA Chatbot Component
Integrates bharatgenai/Param-1-2.9B-Instruct with deterministic domain guards,
expert multi-linguistic knowledge-base grounding, cached resource loading,
and real-time voice microphone input across multiple languages.
"""

import os
import io
import re
import streamlit as st
import soundfile as sf
import speech_recognition as sr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from components.jiva_knowledge import (
    PROJECT_KNOWLEDGE_BASE,
    SYSTEM_INSTRUCTIONS,
    OUT_OF_SCOPE_RESPONSE,
    SUPPORTED_LANGUAGES,
    get_knowledge_response,
)

MODEL_NAME = "bharatgenai/Param-1-2.9B-Instruct"
UNAVAILABLE_RESPONSE = "Ask JIVA is currently unavailable."

# Explicit project keyword patterns for pre-generation domain filtering
PROJECT_KEYWORDS = {
    # Core platform & modules
    "jiva", "lung", "pulmonary", "breath", "exhaled", "respiratory",
    "module", "modules", "platform", "system", "workflow", "pipeline",
    "limitation", "limitations", "disclaimer", "safety", "clinical", "diagnostic",
    # Preprocessing
    "preprocessing", "preprocess", "preprocessed", "imputer", "imputation",
    "yeo-johnson", "robustscaler",
    # VOC Biomarkers & Features
    "voc", "vocs", "volatile", "biomarker", "biomarkers", "compound", "compounds",
    "aldehyde", "aldehydes", "ketone", "ketones", "concentration", "concentrations",
    "ch2o", "c2h4o", "c3h6o", "c4h8o", "c5h10o", "c6h12o", "c7h14o", "c8h16o", "c9h18o",
    "c10h20o", "c11h22o", "c12h24o", "c13h26o", "formaldehyde", "acetaldehyde",
    "acetone", "butanal", "pentanal", "hexanal", "heptanal", "octanal", "nonanal",
    "decanal", "undecanal", "dodecanal", "tridecanal", "acrolein", "benzaldehyde",
    # Classes & Predictions
    "control", "benign", "cancer", "carcinoma", "neoplasm", "malignant", "tumor",
    "nodule", "prediction", "predictions", "predict", "confidence", "probability",
    "probabilities", "risk",
    # Classifiers & ML Algorithms
    "classifier", "classifiers", "regressor", "regressors", "bagging", "random forest",
    "gradient boosting", "extra trees", "lda", "decision tree", "naive bayes", "svm",
    "mlp", "perceptron", "ensemble",
    # Performance & Metrics
    "metric", "metrics", "accuracy", "accurate", "precision", "recall", "sensitivity", "specificity",
    "f1", "mcc", "roc", "auc", "roc-auc", "confusion matrix", "benchmark", "benchmarks",
    "performance", "score", "scores", "rmse", "mae", "r2",
    # Explainability (XAI)
    "shap", "xai", "explainability", "importance", "kernel", "kernelexplainer",
    "attribution", "interpretability",
    # Cardiopulmonary Audio & StethoLM
    "audio", "sound", "sounds", "stetho", "stetholm", "stethoscope", "auscultation",
    "acoustic", "waveform", "spectrogram", "frequency", "wheeze", "wheezes", "wheezing",
    "crackle", "crackles", "rhonchi", "vesicular", "stft", "medgemma",
    # X-ray Analysis
    "xray", "x-ray", "radiograph", "radiographs", "radiological", "radiology",
    "imaging", "chest", "parenchymal", "opacity", "opacities", "consolidation",
    "infiltrate", "costophrenic", "cardiothoracic",
    # Reports & Downloads
    "report", "reports", "pdf", "download", "export", "generate",
}

# Explicit out-of-scope regex patterns that must be rejected immediately
OUT_OF_SCOPE_PATTERNS = [
    r"\b(prime minister|president|election|parliament|government of|cabinet minister)\b",
    r"\b(capital of|population of|tallest mountain|distance to|currency of|geography|world history|history of|ancient history)\b",
    r"\b(write a poem|write a song|tell a joke|write an essay|write a story|fantasy)\b",
    r"\b(crypto|cryptocurrency|bitcoin|ethereum|blockchain|investment advice|stock market|trading)\b",
    r"\b(today'?s news|latest news|breaking news|weather in|weather forecast|temperature in)\b",
    r"\b(write\s+(?:python|java|c\+\+|javascript|typescript)?\s*code|solve\s+(?:a\s+)?dsa|binary tree|leetcode|linked list|sorting algorithm)\b",
    r"\b(unrelated\s+to\s+jiva|not\s+related\s+to\s+jiva|outside\s+jiva|unrelated\s+essay|unrelated\s+disease|unrelated\s+code)\b",
    r"\b(solve\s+math|solve\s+mathematics|math\s+problem|calculus|algebra)\b",
    r"\b(prescribe|prescribing|dosage of|cure for|treat my|medication for|antibiotic for|ibuprofen|paracetamol)\b",
    r"\b(who won the|world cup|olympics|football match|cricket match)\b",
]

# Common conversational in-scope phrasings
CONVERSATIONAL_IN_SCOPE_PHRASES = [
    "what does this result mean",
    "what does my result mean",
    "what does the result mean",
    "what happens after i upload",
    "what happens when i upload",
    "how do i download my report",
    "how do i download",
    "how can i download",
    "how accurate is our model",
    "how accurate is the model",
    "can this diagnose me",
    "can jiva diagnose",
    "can it diagnose",
    "is this a medical",
    "is jiva a medical",
    "what is jiva",
    "what does jiva do",
    "what modules are available",
    "explain our model performance",
    "explain model performance",
    "what do the classes mean",
    "what does cancer prediction mean",
    "why is the jiva used for",
    "why is jiva used for",
    "voc analysis and breath biomarker measurements",
]


def is_project_query(query: str, recent_history: list = None) -> bool:
    """
    Deterministic pre-generation domain guard.
    Returns True if the query relates to the JIVA project; False otherwise.
    Never calls the model for out-of-scope queries.
    """
    if not query or not query.strip():
        return False

    q_lower = query.lower().strip()

    # Step 0: Immediate rejection of queries explicitly stating unrelated topics
    if re.search(r"\b(unrelated|outside\s+jiva|not\s+related)\b", q_lower):
        return False

    # Step 1: Check explicit out-of-scope patterns first
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, q_lower):
            return False

    # Step 2: Handle common greetings & polite entries gracefully
    greeting_patterns = r"\b(hloo|hello|hi|hey|greetings|good morning|good afternoon|good evening|namaste|vanakkam|namaskaram)\b"
    if re.search(greeting_patterns, q_lower) or any(w in q_lower for w in ["హలో", "నమస్కారం", "नमस्ते", "வணக்கம்", "ನಮಸ್ಕಾರ", "hola", "bonjour"]):
        return True

    # Step 3: Check standard conversational in-scope phrases
    for phrase in CONVERSATIONAL_IN_SCOPE_PHRASES:
        if phrase in q_lower:
            return True

    # Step 4: Check multi-lingual keywords (Telugu, Hindi, etc.)
    multilingual_cues = [
        "జేఐవీఏ", "జెఐవిఎ", "శ్వాస", "క్యాన్సర్", "ఆడియో", "ఎక్స్-రే",
        "जेआईवीए", "सांस", "फेफड़े", "कैंसर", "ऑडियो", "एक्स-रे",
    ]
    if any(cue in q_lower for cue in multilingual_cues):
        return True

    # Step 5: Check project keywords in query
    tokens = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", q_lower))
    if "x" in tokens and "ray" in tokens:
        tokens.add("x-ray")
        tokens.add("xray")

    if bool(tokens & PROJECT_KEYWORDS):
        return True

    # Step 6: Substring matching for multi-word or compound keywords
    compound_terms = [
        "volatile organic", "confusion matrix", "feature importance",
        "stetho", "kernel", "robustscaler", "yeo-johnson", "random forest",
        "gradient boosting", "bagging", "cancer recall", "class probab",
        "breath biomarker", "biomarker measurement"
    ]
    if any(term in q_lower for term in compound_terms):
        return True

    # Step 7: Conversational follow-up check using recent chat history
    if recent_history and len(recent_history) > 0:
        follow_up_cues = [
            "why", "how", "what about", "explain more", "elaborate", "tell me more",
            "can you explain", "is that high", "is that low", "is that normal",
            "what does that mean", "which one", "why is that", "could you clarify",
            "what else", "is it good", "is it bad", "what next"
        ]
        if any(q_lower.startswith(cue) or cue in q_lower for cue in follow_up_cues):
            for msg in reversed(recent_history[-3:]):
                content = msg.get("content", "").lower()
                hist_tokens = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", content))
                if bool(hist_tokens & PROJECT_KEYWORDS):
                    return True

    return False


def is_valid_response(response: str) -> bool:
    """
    Deterministic post-generation guard.
    Ensures model response is project-grounded, contains no internal leaks,
    no medical prescriptions, and no confirmed diagnoses.
    """
    if not response or not response.strip():
        return False

    resp_lower = response.lower().strip()

    # Rule 1: No leaking system prompt internals or credentials
    leak_cues = [
        "system_instructions",
        "project_knowledge_base",
        "you are ask jiva, the dedicated ai assistant",
        "rules:\n1. answer only",
        "hf_token",
        "token=",
        "api_key",
    ]
    if any(cue in resp_lower for cue in leak_cues):
        return False

    # Rule 2: No medical prescriptions or treatment orders
    prescribe_cues = [
        "i prescribe",
        "take 500mg",
        "take this medication",
        "i recommend taking",
        "start chemotherapy immediately",
        "you should take antibiotics",
    ]
    if any(cue in resp_lower for cue in prescribe_cues):
        return False

    # Rule 3: No confirmed diagnosis claims
    diagnosis_cues = [
        "you definitely have lung cancer",
        "this confirms you have cancer",
        "i diagnose you with",
        "you have malignant cancer",
    ]
    if any(cue in resp_lower for cue in diagnosis_cues):
        return False

    # Rule 4: No unrelated code block generation
    if "```python" in resp_lower and "def fibonacci" in resp_lower:
        return False
    if "```java" in resp_lower or "```cpp" in resp_lower:
        return False

    return True


def transcribe_audio_query(audio_bytes: bytes, language_code: str = "en-US") -> str:
    """
    Decodes recorded microphone audio bytes and transcribes speech using Google Speech Recognition
    in the designated multi-linguistic language code (e.g. te-IN, hi-IN, en-US, etc.).
    """
    if not audio_bytes or len(audio_bytes) == 0:
        return ""

    r = sr.Recognizer()

    # Strategy 1: Direct AudioFile reading
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = r.record(source)
        transcription = r.recognize_google(audio_data, language=language_code)
        if transcription and transcription.strip():
            return transcription.strip()
    except Exception:
        pass

    # Strategy 2: Resample & convert to standard PCM WAV via soundfile
    try:
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        wav_buf = io.BytesIO()
        sf.write(wav_buf, data, samplerate, format="WAV", subtype="PCM_16")
        wav_buf.seek(0)
        with sr.AudioFile(wav_buf) as source:
            audio_data = r.record(source)
        transcription = r.recognize_google(audio_data, language=language_code)
        if transcription and transcription.strip():
            return transcription.strip()
    except Exception:
        pass

    return ""


@st.cache_resource(show_spinner=False)
def load_jiva_chatbot_model():
    """
    Cached loader for bharatgenai/Param-1-2.9B-Instruct.
    Loads model and tokenizer once per Streamlit server session.
    Fails safely without exposing credentials or internal traces.
    """
    hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
    if not hf_token or not str(hf_token).strip() or "your_huggingface_token_here" in str(hf_token):
        raise RuntimeError("HF_TOKEN credential is not configured.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=True
    )

    is_cuda = torch.cuda.is_available()
    torch_dtype = torch.float16 if is_cuda else torch.float32
    device_map = "auto" if is_cuda else None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    model.eval()

    return tokenizer, model


def generate_jiva_response(user_query: str, recent_messages: list = None, lang_code: str = "en-US") -> str:
    """
    Executes full Ask JIVA pipeline with robust fallback:
    1. Pre-generation domain guard (rejects off-topic queries immediately with exact fallback).
    2. Attempts neural model generation if CUDA is present or sufficient host RAM (>= 6GB) exists.
    3. If neural model is unavailable or hardware-constrained, seamlessly routes to the
       comprehensive, multi-linguistic JIVA Knowledge Base engine.
    Never shows 'Ask JIVA is currently unavailable' for valid project inquiries.
    """
    # Step 1: Pre-generation domain guard
    if not is_project_query(user_query, recent_messages):
        return OUT_OF_SCOPE_RESPONSE

    # Step 2: Check if host hardware can safely accommodate 2.9B parameter neural model
    can_run_model = False
    try:
        import psutil
        if torch.cuda.is_available() or psutil.virtual_memory().available >= 6 * (1024**3):
            can_run_model = True
    except Exception:
        pass

    if can_run_model:
        try:
            tokenizer, model = load_jiva_chatbot_model()

            bounded_history = []
            if recent_messages:
                for msg in recent_messages[-6:]:
                    role = msg.get("role")
                    content = msg.get("content", "").strip()
                    if role in ["user", "assistant"] and content:
                        bounded_history.append({"role": role, "content": content})

            session_enrichment = ""
            active_voc = st.session_state.get("last_voc_prediction")
            if active_voc and isinstance(active_voc, dict):
                pred_class = active_voc.get("predicted_class", "")
                confidence = active_voc.get("confidence", "")
                if pred_class:
                    session_enrichment += f"\nCurrent Active VOC Result: Predicted Class: {pred_class}, Confidence: {confidence}."

            effective_system = SYSTEM_INSTRUCTIONS + (f"\n{session_enrichment}" if session_enrichment else "")

            conversation = [{"role": "system", "content": effective_system}]
            conversation.extend(bounded_history)
            conversation.append({"role": "user", "content": user_query})

            prompt_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                )

            generated_tokens = outputs[0][input_len:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            if is_valid_response(response) and len(response) > 20:
                return response
        except Exception:
            pass

    # Step 3: Seamless expert knowledge-grounded response engine
    return get_knowledge_response(user_query, recent_messages, lang_code=lang_code)



def render_chatbot():
    """
    Renders the Ask JIVA chatbot in the Home page.
    Includes multi-linguistic language selector and real-time voice microphone input.
    Preserves Sarvam-style light layout, topic buttons, and session-state isolation.
    """
    # Isolated session-state initialization
    if "jiva_chat_messages" not in st.session_state:
        st.session_state.jiva_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello. I am the JIVA AI Assistant.\n\n"
                    "You can ask me questions using text or your **Microphone 🎤** in multiple languages:\n"
                    "- **VOC Analysis**: 27 breath biomarker measurements and metabolic patterns\n"
                    "- **Cardiopulmonary Audio**: Auscultation sound analysis and StethoLM reasoning\n"
                    "- **Chest X-ray Inspection**: Radiological screening for pulmonary abnormalities\n"
                    "- **Model Performance**: Accuracy (80.2%), F1 (0.72), Cancer Recall (90.6%)\n"
                    "- **SHAP / XAI**: Feature explainability for patient predictions\n"
                    "- **Diagnostic Reports**: Downloadable in-memory PDF summaries"
                )
            }
        ]

    if "jiva_chat_lang" not in st.session_state:
        st.session_state["jiva_chat_lang"] = "en-US"

    lang_keys = list(SUPPORTED_LANGUAGES.keys())
    current_lang = st.session_state.get("jiva_chat_lang", "en-US")
    selected_lang_idx = lang_keys.index(current_lang) if current_lang in lang_keys else 0

    # Desktop 2-Column Chat Layout
    chat_left, chat_main = st.columns([1, 2.5])

    with chat_left:
        st.markdown("##### Topic Options")

        if st.button("Understanding VOC Results", use_container_width=True, key="topic_voc"):
            user_msg = "Explain VOC analysis and biomarker patterns."
            st.session_state.jiva_chat_messages.append({"role": "user", "content": user_msg})
            reply = generate_jiva_response(user_msg, st.session_state.jiva_chat_messages[:-1], lang_code=current_lang)
            st.session_state.jiva_chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.button("Audio Analysis", use_container_width=True, key="topic_audio"):
            user_msg = "How does Cardiopulmonary Audio analysis work?"
            st.session_state.jiva_chat_messages.append({"role": "user", "content": user_msg})
            reply = generate_jiva_response(user_msg, st.session_state.jiva_chat_messages[:-1], lang_code=current_lang)
            st.session_state.jiva_chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.button("X-ray Analysis", use_container_width=True, key="topic_xray"):
            user_msg = "Explain the X-ray analysis pipeline."
            st.session_state.jiva_chat_messages.append({"role": "user", "content": user_msg})
            reply = generate_jiva_response(user_msg, st.session_state.jiva_chat_messages[:-1], lang_code=current_lang)
            st.session_state.jiva_chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.button("Model Performance", use_container_width=True, key="topic_perf"):
            user_msg = "What are the model performance metrics?"
            st.session_state.jiva_chat_messages.append({"role": "user", "content": user_msg})
            reply = generate_jiva_response(user_msg, st.session_state.jiva_chat_messages[:-1], lang_code=current_lang)
            st.session_state.jiva_chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.button("Explainability", use_container_width=True, key="topic_shap"):
            user_msg = "How does SHAP explainability work?"
            st.session_state.jiva_chat_messages.append({"role": "user", "content": user_msg})
            reply = generate_jiva_response(user_msg, st.session_state.jiva_chat_messages[:-1], lang_code=current_lang)
            st.session_state.jiva_chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()

        st.markdown("<hr style='margin: 1.2rem 0; border-color: #e8e8e8;'>", unsafe_allow_html=True)
        st.markdown("##### 🌐 Language / భాష")

        lang_display_options = [f"{SUPPORTED_LANGUAGES[k]['flag']} {SUPPORTED_LANGUAGES[k]['label']}" for k in lang_keys]
        chosen_lang_str = st.selectbox(
            "Language",
            options=lang_display_options,
            index=selected_lang_idx,
            key="jiva_lang_selector",
            label_visibility="collapsed"
        )
        new_lang_code = lang_keys[lang_display_options.index(chosen_lang_str)]
        if new_lang_code != st.session_state["jiva_chat_lang"]:
            st.session_state["jiva_chat_lang"] = new_lang_code
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear Chat", use_container_width=True, key="clear_chat_history"):
            st.session_state.jiva_chat_messages = []
            st.rerun()

    with chat_main:
        active_lang_meta = SUPPORTED_LANGUAGES.get(st.session_state["jiva_chat_lang"], SUPPORTED_LANGUAGES["en-US"])
        active_lang_code = st.session_state["jiva_chat_lang"]
        active_lang_name = active_lang_meta["name"]

        chat_container = st.container()
        with chat_container:
            for message in st.session_state.jiva_chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Voice Microphone Input Section
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #e8e8e8; border-radius: 12px; padding: 10px 14px; margin-top: 1rem; margin-bottom: 0.8rem; display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.2rem;">🎤</span>
                    <span style="font-weight: 600; font-size: 0.95rem; color: #252525;">Voice Microphone ({active_lang_meta['flag']} {active_lang_meta['label']})</span>
                </div>
                <span style="font-size: 0.82rem; color: #707070;">Speak or record your question below</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.components.v1.html(
            f"""
            <div style="font-family: system-ui, -apple-system, sans-serif; display: flex; align-items: center; gap: 10px; padding: 4px 0;">
                <button id="web-mic-btn" onclick="startWebSpeech()" style="background: #4b63e6; color: white; border: none; border-radius: 8px; padding: 8px 14px; font-size: 13px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <span id="mic-icon">🎤</span>
                    <span id="mic-label">Live Mic: Speak in {active_lang_name}</span>
                </button>
                <div id="mic-transcript" style="font-size: 12px; color: #555; background: #f4fbfd; border: 1px solid #cfeaf6; padding: 6px 12px; border-radius: 6px; display: none; flex-grow: 1;"></div>
            </div>
            <script>
                var recognizing = false;
                var recognition = null;
                if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
                    var SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
                    recognition = new SpeechRec();
                    recognition.continuous = false;
                    recognition.interimResults = true;
                    recognition.lang = '{active_lang_code}';

                    recognition.onstart = function() {{
                        recognizing = true;
                        document.getElementById('mic-label').innerText = 'Listening... Speak in {active_lang_name}';
                        document.getElementById('web-mic-btn').style.background = '#e63946';
                        var t = document.getElementById('mic-transcript');
                        t.style.display = 'block';
                        t.innerText = 'Listening...';
                    }};

                    recognition.onresult = function(event) {{
                        var transcript = '';
                        for (var i = event.resultIndex; i < event.results.length; ++i) {{
                            transcript += event.results[i][0].transcript;
                        }}
                        var t = document.getElementById('mic-transcript');
                        t.innerText = 'Recognized: "' + transcript + '" (copied to input/clipboard)';
                        try {{
                            var parentTextarea = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
                            if (parentTextarea) {{
                                parentTextarea.value = transcript;
                                parentTextarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            }}
                        }} catch(e) {{}}
                        if (navigator.clipboard) {{
                            navigator.clipboard.writeText(transcript);
                        }}
                    }};

                    recognition.onerror = function(event) {{
                        recognizing = false;
                        document.getElementById('mic-label').innerText = 'Live Mic: Speak in {active_lang_name}';
                        document.getElementById('web-mic-btn').style.background = '#4b63e6';
                    }};

                    recognition.onend = function() {{
                        recognizing = false;
                        document.getElementById('mic-label').innerText = 'Live Mic: Speak in {active_lang_name}';
                        document.getElementById('web-mic-btn').style.background = '#4b63e6';
                    }};
                }}

                function startWebSpeech() {{
                    if (!recognition) {{
                        alert('Live Web Speech API is not supported on this browser. Please use the Audio Recorder below.');
                        return;
                    }}
                    if (recognizing) {{
                        recognition.stop();
                    }} else {{
                        recognition.lang = '{active_lang_code}';
                        recognition.start();
                    }}
                }}
            </script>
            """,
            height=48
        )

        audio_clip = st.audio_input(
            f"Record voice query ({active_lang_name})",
            key="jiva_audio_mic_input",
            label_visibility="collapsed"
        )


        if audio_clip is not None:
            audio_bytes = audio_clip.read()
            audio_hash = hash(audio_bytes)
            if st.session_state.get("last_processed_audio_hash") != audio_hash:
                st.session_state["last_processed_audio_hash"] = audio_hash
                with st.spinner(f"Transcribing voice query in {active_lang_name}..."):
                    transcribed = transcribe_audio_query(audio_bytes, language_code=active_lang_code)

                if transcribed and transcribed.strip():
                    voice_user_msg = f"🎤 [{active_lang_name}]: {transcribed.strip()}"
                    st.session_state.jiva_chat_messages.append({"role": "user", "content": voice_user_msg})

                    reply = generate_jiva_response(
                        transcribed.strip(),
                        st.session_state.jiva_chat_messages[:-1],
                        lang_code=active_lang_code
                    )

                    st.session_state.jiva_chat_messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                else:
                    st.info(f"Could not clearly transcribe the audio in {active_lang_name}. Please try speaking clearly or type your question below.")

        # Text Chat Input
        placeholder_text = f"Ask JIVA in {active_lang_name} or English..."
        if user_input := st.chat_input(placeholder_text):
            st.session_state.jiva_chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            bot_reply = generate_jiva_response(
                user_input,
                st.session_state.jiva_chat_messages[:-1],
                lang_code=active_lang_code
            )

            st.session_state.jiva_chat_messages.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
