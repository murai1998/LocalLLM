#!/usr/bin/env python3
"""Streamlit multimodal chat with Agent and Translate modes."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from localllm.agents import build_agent_graph, discover_skills, invoke_agent
from localllm.chat import ChatEngine, UserTurn
from localllm.config import get_settings
from localllm.media.attachments import (
    AppMode,
    attachment_kind,
    build_multimodal_content,
    format_user_error,
    prepare_agent_context,
    prepare_chat_turn,
    validate_extension,
)
from localllm.secrets import apply_hf_token

UPLOAD_ROOT = Path(tempfile.gettempdir()) / "localllm" / "streamlit_uploads"
APP_MODES = ["Chat", "Agent", "Translate"]

CHAT_UPLOADER_TYPES = ["png", "jpg", "jpeg", "webp", "gif", "pdf", "wav", "txt", "md", "docx"]
AGENT_UPLOADER_TYPES = [
    "png", "jpg", "jpeg", "webp", "gif", "pdf",
    "wav", "mp3", "m4a", "flac", "ogg", "webm", "aac",
    "txt", "md", "docx", "csv", "json", "yaml", "yml", "html", "htm", "log", "xml",
]


def _is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


@st.cache_resource
def get_engine() -> ChatEngine:
    apply_hf_token()
    return ChatEngine()


@st.cache_resource
def get_agent_graph(skill_names: tuple[str, ...]):
    from localllm.agents.skills import resolve_skills
    return build_agent_graph(
        autostart_server=True,
        skills=resolve_skills(list(skill_names)),
    )


def _format_agent_trace(messages: list) -> list[dict]:
    trace: list[dict] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                trace.append({
                    "kind": "tool_call",
                    "title": f"🔍 Tool Call: {tc['name']}",
                    "body": str(tc.get("args", {})),
                })
        elif isinstance(msg, ToolMessage):
            trace.append({
                "kind": "tool_result",
                "title": f"📥 Result: {msg.name}",
                "body": msg.content[:2000] + "..." if len(msg.content) > 2000 else msg.content,
            })
    return trace


def _session_upload_dir() -> Path:
    if "upload_session_id" not in st.session_state:
        st.session_state.upload_session_id = uuid.uuid4().hex
    path = UPLOAD_ROOT / st.session_state.upload_session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


DEFAULT_ENABLED_SKILLS = {"internet-access", "media-convert"}


def _skill_checkbox_key(skill_name: str) -> str:
    return f"skill_cb_{skill_name}"


def _init_skill_defaults(available_skills: list) -> None:
    for skill in available_skills:
        key = _skill_checkbox_key(skill.name)
        if key not in st.session_state:
            st.session_state[key] = skill.name in DEFAULT_ENABLED_SKILLS


def _selected_skills(available_skills: list) -> list[str]:
    return [
        skill.name
        for skill in available_skills
        if st.session_state.get(_skill_checkbox_key(skill.name), False)
    ]


def _render_skill_checkboxes(available_skills: list) -> list[str]:
    st.markdown("**Agent capabilities**")
    st.caption("Enable the skills Sage may use for this session.")
    for skill in available_skills:
        label = skill.name.replace("-", " ").title()
        st.checkbox(
            label,
            help=skill.description or "No description",
            key=_skill_checkbox_key(skill.name),
        )
    selected = _selected_skills(available_skills)
    if not selected:
        st.warning("No skills enabled — Sage will only have basic file tools.")
    return selected


def _default_app_mode() -> str:
    env_mode = os.environ.get("LOCALLLM_STREAMLIT_MODE", "").strip().lower()
    if env_mode == "translate":
        return "Translate"
    return "Chat"


def _init_session_state() -> None:
    if "ui_messages" not in st.session_state:
        st.session_state.ui_messages = []
    if "pending_attachments" not in st.session_state:
        st.session_state.pending_attachments = []
    if "attachment_errors" not in st.session_state:
        st.session_state.attachment_errors = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = _default_app_mode()
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []


def _save_upload(uploaded, directory: Path) -> Path:
    dest = directory / uploaded.name
    dest.write_bytes(uploaded.getvalue())
    return dest


def _remove_attachment(attach_id: str) -> None:
    st.session_state.pending_attachments = [
        item for item in st.session_state.pending_attachments if item["id"] != attach_id
    ]


def _stage_uploads(uploads: list[Any], mode: AppMode) -> bool:
    """Stage new uploads only; returns True if anything was added."""
    if not uploads:
        return False
    upload_dir = _session_upload_dir()
    known = {item["name"] for item in st.session_state.pending_attachments}
    new_errors: list[str] = []
    added = False

    for uploaded in uploads:
        if uploaded.name in known:
            continue
        err = validate_extension(uploaded.name, mode)
        if err:
            new_errors.append(err)
            continue
        path = _save_upload(uploaded, upload_dir)
        st.session_state.pending_attachments.append({
            "id": uuid.uuid4().hex,
            "name": uploaded.name,
            "path": str(path),
            "suffix": path.suffix.lower(),
            "kind": attachment_kind(path.suffix.lower()),
            "sent": False,
        })
        added = True

    if new_errors:
        st.session_state.attachment_errors = new_errors
    return added


def _render_attachment_errors() -> None:
    for err in st.session_state.get("attachment_errors", []):
        st.error(err, icon="⚠️")
    st.session_state.attachment_errors = []


def _render_attachment_item(item: dict[str, Any], *, in_sidebar: bool) -> None:
    path = Path(item["path"])
    kind = item.get("kind", "file")
    cols = st.columns([4, 1]) if in_sidebar else st.columns([6, 1])
    with cols[0]:
        st.markdown(f"**{item['name']}**")
        status = "in conversation" if item.get("sent") else "sends with next message"
        st.caption(f"{kind} · {status}")
        if kind == "image" and path.is_file():
            st.image(str(path), width=120)
        elif kind == "audio" and path.is_file():
            st.audio(str(path))
        elif kind == "pdf":
            st.caption("PDF — text or page images")
        elif kind == "document":
            st.caption("Text document")
    with cols[1]:
        st.button(
            "✕",
            key=f"detach_{item['id']}_{'sb' if in_sidebar else 'main'}",
            help="Remove attachment",
            on_click=_remove_attachment,
            args=(item["id"],),
        )


def _render_pending_attachments(*, in_sidebar: bool) -> None:
    items = st.session_state.pending_attachments
    if not items:
        return
    label = "**Session attachments**" if in_sidebar else "**Attachments for next message**"
    st.markdown(label)
    for item in items:
        _render_attachment_item(item, in_sidebar=in_sidebar)
    if in_sidebar and st.button("Clear all attachments", key="clear_all_attachments"):
        st.session_state.pending_attachments = []
        st.session_state.uploader_key += 1
        st.rerun()


def _attachments_for_next_message() -> list[dict[str, Any]]:
    return [item for item in st.session_state.pending_attachments if not item.get("sent")]


def _attachment_paths_for_next_message() -> list[Path]:
    return [Path(item["path"]) for item in _attachments_for_next_message()]


def _mark_attachments_sent(paths: list[Path]) -> None:
    sent_paths = {str(path) for path in paths}
    for item in st.session_state.pending_attachments:
        if item["path"] in sent_paths:
            item["sent"] = True


def _render_attachment_sidebar(mode: AppMode) -> None:
    uploader_types = CHAT_UPLOADER_TYPES if mode == "chat" else AGENT_UPLOADER_TYPES
    caption = (
        "Chat: images, PDF, **WAV audio**, documents."
        if mode == "chat"
        else "Agent: broader formats; m4a/mp3 auto-converted to WAV (needs imageio-ffmpeg)."
    )
    st.caption(
        caption
        + " Attachments stay in this pane for the session; remove with ✕ or Reset."
    )
    uploads = st.file_uploader(
        "Attach files (sent with next message)",
        accept_multiple_files=True,
        key=f"attach_{mode}_{st.session_state.uploader_key}",
        type=uploader_types,
    )
    if _stage_uploads(uploads or [], mode):
        st.session_state.uploader_key += 1
        st.rerun()
    _render_attachment_errors()
    _render_pending_attachments(in_sidebar=True)


def run_app() -> None:
    st.set_page_config(page_title="LocalLLM — Sage", layout="wide")
    _init_session_state()

    st.title("🧘 Sage — Gemma 4 12B Local Assistant")
    st.caption("Chat · Agent · Voice Translate | llama.cpp")

    translate_params = None

    with st.sidebar:
        st.subheader("Mode")
        mode_index = APP_MODES.index(st.session_state.app_mode) if st.session_state.app_mode in APP_MODES else 0
        st.session_state.app_mode = st.radio(
            "App mode",
            options=APP_MODES,
            index=mode_index,
            horizontal=True,
            label_visibility="collapsed",
        )

        if st.session_state.app_mode == "Agent":
            available = list(discover_skills())
            if available:
                _init_skill_defaults(available)
                st.session_state.enabled_skills = _render_skill_checkboxes(available)
            else:
                st.session_state.enabled_skills = []
                st.caption("No skills installed in skills/*/SKILL.md")
            _render_attachment_sidebar("agent")

        elif st.session_state.app_mode == "Chat":
            _render_attachment_sidebar("chat")

        elif st.session_state.app_mode == "Translate":
            from apps.streamlit_translate import render_translate_sidebar
            translate_params = render_translate_sidebar()

        st.subheader("Session")
        if st.button("Reset conversation"):
            if st.session_state.app_mode == "Chat":
                get_engine().reset()
                st.session_state.ui_messages = []
            elif st.session_state.app_mode == "Agent":
                st.session_state.agent_messages = []
            elif st.session_state.app_mode == "Translate":
                for key in ("transcript", "translation", "tts_audio", "translate_result", "recorded_audio"):
                    st.session_state.pop(key, None)
            st.session_state.pending_attachments = []
            st.session_state.attachment_errors = []
            st.session_state.uploader_key += 1
            st.rerun()

    if st.session_state.app_mode == "Agent":
        _run_agent_ui()
        return

    if st.session_state.app_mode == "Translate":
        from apps.streamlit_translate import run_translate_ui
        if translate_params is None:
            translate_params = ("auto", "es", "professional", None)
        source_lang, target_lang, tone, voice_id = translate_params
        run_translate_ui(
            source_lang=source_lang,
            target_lang=target_lang,
            tone=tone,
            voice_id=voice_id,
        )
        return

    _run_chat_ui()


def _run_agent_ui() -> None:
    _render_pending_attachments(in_sidebar=False)

    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"] if isinstance(msg["content"], str) else "(multimodal message)")
            for step in msg.get("trace", []):
                with st.expander(step["title"], expanded=False):
                    st.code(step["body"])

    prompt = st.chat_input("Ask Sage anything...")
    if not prompt or not prompt.strip():
        return

    user_prompt = prompt.strip()
    pending = _attachments_for_next_message()
    attachment_names = [item["name"] for item in pending]
    display = user_prompt
    if attachment_names:
        display += "\n\n_Attachments: " + ", ".join(attachment_names) + "_"

    engine = get_engine()
    try:
        attach_paths = _attachment_paths_for_next_message()
        user_text, image_paths, audio_path = prepare_agent_context(
            user_prompt,
            attach_paths,
        )
        user_content = build_multimodal_content(
            user_text,
            image_paths=image_paths,
            audio_path=audio_path,
            client=engine.client,
        )
    except Exception as exc:
        st.error(format_user_error(exc))
        return

    st.session_state.agent_messages.append({"role": "user", "content": display})

    skill_names = tuple(sorted(st.session_state.enabled_skills))
    graph = get_agent_graph(skill_names)

    with st.chat_message("assistant"):
        with st.spinner("🧘 Sage is thinking and using tools..."):
            try:
                result = invoke_agent(
                    graph,
                    {"messages": [HumanMessage(content=user_content)]},
                )
                trace = _format_agent_trace(result["messages"])

                final_reply = ""
                for m in reversed(result["messages"]):
                    if not isinstance(m, AIMessage) or not m.content:
                        continue
                    if getattr(m, "tool_calls", None):
                        continue
                    if "<|tool_call>" in m.content:
                        continue
                    final_reply = m.content
                    break
                if not final_reply:
                    final_reply = "(No final answer received — try rephrasing or use --verbose in CLI)"

            except Exception as exc:
                trace = []
                final_reply = format_user_error(exc)

            if trace:
                st.caption("**Agent Steps:**")
                for step in trace:
                    with st.expander(step["title"], expanded=False):
                        st.code(step["body"])

            st.markdown(final_reply)

    st.session_state.agent_messages.append({
        "role": "assistant",
        "content": final_reply,
        "trace": trace,
    })
    _mark_attachments_sent(attach_paths)
    st.rerun()


def _run_chat_ui() -> None:
    _render_pending_attachments(in_sidebar=False)

    for msg in st.session_state.ui_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for img in msg.get("images", []):
                if Path(img).is_file():
                    st.image(img)

    prompt = st.chat_input("Message…")
    if not prompt or not prompt.strip():
        return

    user_prompt = prompt.strip()
    pending = _attachments_for_next_message()
    attach_paths = _attachment_paths_for_next_message()
    try:
        user_text, image_paths, audio_path = prepare_chat_turn(
            user_prompt,
            attach_paths,
        )
    except Exception as exc:
        st.error(format_user_error(exc))
        return

    if not user_text and not image_paths and not audio_path:
        st.warning("Nothing to send.")
        return

    attachment_names = [item["name"] for item in pending]
    display = user_text or "(multimodal message)"
    if attachment_names:
        display += "\n\n_Attachments: " + ", ".join(attachment_names) + "_"

    st.session_state.ui_messages.append({
        "role": "user",
        "content": display,
        "images": [str(p) for p in image_paths],
    })

    engine = get_engine()
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = engine.send(
                    UserTurn(text=user_text, image_paths=image_paths, audio_path=audio_path)
                )
            except Exception as exc:
                reply = format_user_error(exc)
            st.markdown(reply)

    st.session_state.ui_messages.append({"role": "assistant", "content": reply})
    _mark_attachments_sent(attach_paths)
    st.rerun()


if _is_streamlit_runtime():
    run_app()