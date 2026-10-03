# ============================================================
# AI-ASSISTED MULTI-LABEL CHEST X-RAY INTERPRETATION
# Streamlit Application
# ============================================================

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms

from sentence_transformers import SentenceTransformer

from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Chest X-ray Assistant",
    page_icon="🩻",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\yuvan\Desktop\ML_projects\Chest_Xray_AI"
)

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vgg19_best.pth"

RAG_INDEX_PATH = (
    PROJECT_ROOT
    / "rag"
    / "v2"
    / "vector_store"
    / "rag_v2.index"
)

RAG_METADATA_PATH = (
    PROJECT_ROOT
    / "rag"
    / "v2"
    / "metadata"
    / "rag_v2_metadata.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LABELS
# ============================================================

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]


# ============================================================
# MODEL
# ============================================================

class VGG19Grayscale(nn.Module):

    def __init__(self, num_classes=14):

        super().__init__()

        backbone = models.vgg19(weights=None)

        features = backbone.features

        original_conv = features[0]

        features[0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


@st.cache_resource
def load_model():

    model = VGG19Grayscale(
        num_classes=14
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Required for Grad-CAM
    for module in model.modules():

        if isinstance(module, nn.ReLU):
            module.inplace = False

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485],
        std=[0.229]
    )
])


# ============================================================
# RAG
# ============================================================

@st.cache_resource
def load_rag():

    index = faiss.read_index(
        str(RAG_INDEX_PATH)
    )

    with open(
        RAG_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return (
        index,
        metadata,
        embedding_model
    )


# ============================================================
# GEMINI
# ============================================================

@st.cache_resource
def load_gemini():

    api_key = None

    import os

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY environment variable not found."
        )

    client = genai.Client(
        api_key=api_key
    )

    return client


MODEL_NAME = "gemini-3.1-flash-lite"

gemini_config = types.GenerateContentConfig(
    temperature=0.2,
    max_output_tokens=500
)


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_v2(
    query,
    top_k=5
):

    rag_index, rag_metadata, embedding_model = load_rag()

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = rag_index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        metadata = rag_metadata[int(idx)]

        results.append({

            "rank": rank,

            "score": float(score),

            "chunk_id": metadata["chunk_id"],

            "finding": metadata["finding"],

            "section": metadata["section"],

            "title": metadata["title"],

            "url": metadata["url"],

            "text": metadata["text"]

        })

    return results


# ============================================================
# FINDING-SPECIFIC RAG
# ============================================================

def get_finding_rag_context(
    finding,
    top_k=5
):

    query_templates = {

        "Atelectasis":
            "What is atelectasis and what are its relevant features on a chest X-ray?",

        "Cardiomegaly":
            "What is cardiomegaly and what are its relevant features on a chest X-ray?",

        "Effusion":
            "What is pleural effusion and what are its relevant features on a chest X-ray?",

        "Infiltration":
            "What does the Infiltration label mean in the ChestX-ray14 dataset, and how should it be interpreted?",

        "Mass":
            "What is a lung mass and what are its relevant imaging features?",

        "Nodule":
            "What is a lung nodule and how is it detected and evaluated on chest imaging?",

        "Pneumonia":
            "What is pneumonia and what are its relevant features on a chest X-ray?",

        "Pneumothorax":
            "What is pneumothorax and how does it appear on a chest X-ray?",

        "Consolidation":
            "What is lung consolidation and what does it indicate on a chest X-ray?",

        "Edema":
            "What is pulmonary edema and what are its relevant features on chest imaging?",

        "Emphysema":
            "What is emphysema and what are its relevant features on chest imaging?",

        "Fibrosis":
            "What is pulmonary fibrosis and what are its relevant imaging features?",

        "Pleural_Thickening":
            "What is pleural thickening and how does it appear on chest imaging?",

        "Hernia":
            "What is a diaphragmatic hernia and what are its relevant imaging features?"
    }

    query = query_templates[finding]

    retrieved = retrieve_v2(
        query,
        top_k=top_k
    )

    primary_evidence = []
    general_evidence = []

    for item in retrieved:

        if item["finding"] == finding:

            primary_evidence.append(item)

        elif item["finding"] == "General":

            general_evidence.append(item)

    if primary_evidence:

        evidence_status = "SUFFICIENT"

    elif general_evidence:

        evidence_status = "LIMITED"

    else:

        evidence_status = "INSUFFICIENT"

    return {

        "query": query,

        "evidence_status": evidence_status,

        "primary_evidence": primary_evidence,

        "general_evidence": general_evidence

    }


# ============================================================
# GEMINI INITIAL INTERPRETATION
# ============================================================

def generate_initial_interpretation(
    finding,
    probability,
    gradcam_description,
    rag_context
):

    client = load_gemini()

    evidence = (
        rag_context["primary_evidence"]
        + rag_context["general_evidence"]
    )

    evidence_text = "\n\n".join(

        f"""
CHUNK ID: {item['chunk_id']}
TITLE: {item['title']}
FINDING: {item['finding']}

CONTENT:
{item['text']}
"""
        for item in evidence
    )

    prompt = f"""
You are an evidence-grounded medical information assistant
supporting an academic chest X-ray AI system.

CNN prediction:
Finding: {finding}
Probability: {probability:.4f}

Grad-CAM:
{gradcam_description}

Retrieved medical evidence:
{evidence_text}

Evidence status:
{rag_context['evidence_status']}

Generate a concise structured interpretation.

Important:
- Do not change the CNN prediction.
- Do not treat probability as a diagnosis.
- Explain what the finding generally means.
- Explain the Grad-CAM only as model attribution.
- Mention relevant limitations.
- ChestX-ray14 labels are research dataset annotations.
- Do not invent citations.
- Return JSON only.

Format:

{{
    "medical_context": "string",
    "interpretation": "string",
    "limitations": [
        "string",
        "string"
    ],
    "citations": [
        {{
            "chunk_id": "string",
            "title": "string"
        }}
    ]
}}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=gemini_config
    )

    raw = response.text.strip()

    if raw.startswith("```"):

        raw = raw.replace(
            "```json",
            ""
        )

        raw = raw.replace(
            "```",
            ""
        )

        raw = raw.strip()

    return json.loads(raw)


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image
):

    input_tensor = (
        inference_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    input_tensor.requires_grad_(True)

    # First prediction
    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )[0].cpu().numpy()

    top_index = int(
        np.argmax(probabilities)
    )

    finding = LABELS[top_index]

    # Hooks
    activations = None
    gradients = None

    def forward_hook(
        module,
        input,
        output
    ):

        nonlocal activations

        activations = output

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        nonlocal gradients

        gradients = grad_output[0]

    target_layer = model.features[34]

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )

    model.zero_grad(
        set_to_none=True
    )

    logits = model(
        input_tensor
    )

    target_logit = logits[
        0,
        top_index
    ]

    target_logit.backward()

    forward_handle.remove()
    backward_handle.remove()

    activation = activations[0]

    gradient = gradients[0]

    weights = gradient.mean(
        dim=(1, 2)
    )

    cam = torch.sum(
        weights[:, None, None]
        * activation,
        dim=0
    )

    cam = torch.relu(cam)

    cam = (
        cam.detach()
        .cpu()
        .numpy()
    )

    if cam.max() > 0:

        cam = (
            cam / cam.max()
        )

    cam_tensor = (
        torch.from_numpy(cam)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    cam_resized = F.interpolate(
        cam_tensor,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    )

    cam_resized = (
        cam_resized
        .squeeze()
        .numpy()
    )

    return (
        probabilities,
        finding,
        top_index,
        cam_resized
    )


# ============================================================
# FULL INITIAL ANALYSIS
# ============================================================

def run_analysis(
    image
):

    model = load_model()

    probabilities, finding, top_index, cam = (
        generate_gradcam(
            model,
            image
        )
    )

    probability = float(
        probabilities[top_index]
    )

    predictions = {

        label: float(prob)

        for label, prob in zip(
            LABELS,
            probabilities
        )
    }

    gradcam_description = (
        f"Grad-CAM highlights image regions "
        f"that contributed to the {finding} "
        f"prediction. The heatmap represents "
        f"model attribution and should not be "
        f"interpreted as direct anatomical or "
        f"diagnostic evidence."
    )

    rag_context = get_finding_rag_context(
        finding
    )

    llm = generate_initial_interpretation(
        finding,
        probability,
        gradcam_description,
        rag_context
    )

    return {

        "predictions": predictions,

        "finding": finding,

        "probability": probability,

        "cam": cam,

        "medical_context":
            llm["medical_context"],

        "interpretation":
            llm["interpretation"],

        "limitations":
            llm["limitations"],

        "citations":
            llm["citations"],

        "rag_status":
            rag_context["evidence_status"],

        "rag_context":
            rag_context
    }


# ============================================================
# DIRECT GEMINI FOLLOW-UP
# ============================================================

def ask_gemini_followup(
    question,
    finding,
    probability,
    initial_interpretation
):

    client = load_gemini()

    prompt = f"""
You are the conversational medical-information assistant
for an academic chest X-ray AI system.

Initial AI analysis:

Finding:
{finding}

CNN probability:
{probability:.4f}

Initial interpretation:
{initial_interpretation}

User question:
{question}

Instructions:

1. Answer the question directly.
2. Use the initial result only as context.
3. Do not change or override the CNN prediction.
4. Do not claim the X-ray alone establishes a diagnosis.
5. Provide general medical information.
6. Do not create a personalized treatment plan.
7. If discussing medications, treatment, prognosis,
   or recovery, explain that individual management
   depends on clinical evaluation.
8. Do not invent precise recovery timelines.

Return only the answer text.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=gemini_config
    )

    return response.text.strip()


# ============================================================
# SESSION STATE
# ============================================================

if "analysis" not in st.session_state:

    st.session_state.analysis = None

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ============================================================
# HEADER
# ============================================================

st.title(
    "🩻 AI-Assisted Chest X-ray Assistant"
)

st.caption(
    "Multi-label VGG-19 + Grad-CAM + RAG-grounded "
    "medical interpretation + Gemini follow-up"
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# ANALYZE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("L")

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.subheader(
            "Uploaded X-ray"
        )

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Analysis"
        )

        if st.button(
            "🔍 Analyze X-ray",
            type="primary"
        ):

            with st.spinner(
                "Running VGG-19 → Grad-CAM → RAG → Gemini..."
            ):

                st.session_state.analysis = (
                    run_analysis(image)
                )

                st.session_state.chat_history = []

            st.success(
                "Analysis completed!"
            )


# ============================================================
# DISPLAY ANALYSIS
# ============================================================

analysis = st.session_state.analysis

if analysis is not None:

    st.divider()

    st.header(
        "AI Analysis"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.metric(
            "Top Finding",
            analysis["finding"]
        )

    with col2:

        st.metric(
            "Model Probability",
            f"{analysis['probability']:.2%}"
        )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    if "show_predictions" not in st.session_state:
        st.session_state.show_predictions = False

    col1, col2 = st.columns([5, 1])

    with col2:
        button_label = (
            "Hide 14-labels"
            if st.session_state.show_predictions
            else "View 14-labels"
        )

        if st.button(
            button_label,
            use_container_width=True
        ):
            st.session_state.show_predictions = (
                not st.session_state.show_predictions
            )
            st.rerun()

    if st.session_state.show_predictions:

        st.subheader(
            "14-label Predictions"
        )

        sorted_predictions = sorted(
            analysis["predictions"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for label, probability in sorted_predictions:

            st.write(
                f"**{label}** — "
                f"{probability:.2%}"
            )

            st.progress(
                min(
                    probability,
                    1.0
                )
            )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    st.subheader(
        "Grad-CAM"
    )

    cam = analysis["cam"]

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.image(
            image,
            caption="Original X-ray",
            use_container_width=True
        )

    with col2:

        st.image(
            cam,
            caption="Model attribution heatmap",
            use_container_width=True
        )

    st.info(
        "Grad-CAM shows image regions that contributed "
        "to the model prediction. It is not direct "
        "anatomical or diagnostic evidence."
    )

    # --------------------------------------------------------
    # Medical context
    # --------------------------------------------------------

    st.subheader(
        "Medical Context"
    )

    st.write(
        analysis["medical_context"]
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.subheader(
        "AI Interpretation"
    )

    st.write(
        analysis["interpretation"]
    )

    # --------------------------------------------------------
    # RAG status
    # --------------------------------------------------------

    st.subheader(
        "Evidence"
    )

    if analysis["rag_status"] == "SUFFICIENT":

        st.success(
            "Finding-specific evidence retrieved"
        )

    elif analysis["rag_status"] == "LIMITED":

        st.warning(
            "Only limited supporting evidence was retrieved."
        )

    else:

        st.error(
            "Insufficient supporting evidence was retrieved."
        )

    # --------------------------------------------------------
    # Limitations
    # --------------------------------------------------------

    st.subheader(
        "Limitations"
    )

    for limitation in analysis["limitations"]:

        st.write(
            f"• {limitation}"
        )

    # --------------------------------------------------------
    # Citations
    # --------------------------------------------------------

    st.subheader(
        "Sources"
    )

    rag_metadata = load_rag()[1]

    metadata_by_id = {

        item["chunk_id"]: item

        for item in rag_metadata
    }

    for citation in analysis["citations"]:

        chunk_id = citation["chunk_id"]

        source = metadata_by_id.get(
            chunk_id
        )

        if source:

            st.markdown(
                f"**{source['title']}**  \n"
                f"{source['url']}"
            )

    # ========================================================
    # FOLLOW-UP CHAT
    # ========================================================

    st.divider()

    st.header(
        "💬 Ask a Follow-up Question"
    )

    st.caption(
        "Follow-up questions are answered directly "
        "by Gemini using the context of the initial analysis."
    )

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    question = st.chat_input(
        "Ask something about the result..."
    )

    if question:

        st.session_state.chat_history.append({

            "role": "user",

            "content": question

        })

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Gemini is thinking..."
            ):

                answer = ask_gemini_followup(

                    question=question,

                    finding=analysis["finding"],

                    probability=analysis["probability"],

                    initial_interpretation=
                        analysis["interpretation"]
                )

            st.write(
                answer
            )

            st.session_state.chat_history.append({

                "role": "assistant",

                "content": answer

            })


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Academic AI research prototype. "
    "Model predictions and generated information "
    "should not be treated as a medical diagnosis."
) 