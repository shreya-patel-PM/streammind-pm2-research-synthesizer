import streamlit as st
import os
import json
import re
from dotenv import load_dotenv
from io import BytesIO
import docx
import pypdf
import anthropic

# Load API key
load_dotenv()

try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv("ANTHROPIC_API_KEY")


def parse_uploaded_file(uploaded_file):
    filename = uploaded_file.name
    extension = filename.split(".")[-1].lower()

    try:
        if extension == "txt":
            content = uploaded_file.read().decode("utf-8")
            return content

        elif extension == "pdf":
            pdf_reader = pypdf.PdfReader(BytesIO(uploaded_file.read()))
            text_parts = []
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())
            return "\n\n".join(text_parts)

        elif extension == "docx":
            doc = docx.Document(BytesIO(uploaded_file.read()))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)

        else:
            raise ValueError("Unsupported file type: ." + extension)

    except Exception as e:
        raise ValueError("Could not parse " + filename + ": " + str(e))


def build_synthesis_request(research_topic, transcripts):
    system_prompt = (
        "You are a senior user researcher synthesizing themes across multiple "
        "user interviews for a product team. Your job is to find patterns that "
        "matter, ground them in verbatim quotes, and surface what the team "
        "should do about it.\n\n"
        "Output ONLY valid JSON. No markdown fences. No prose before or after. "
        "Your entire response must begin with { and end with }.\n\n"
        "Schema:\n"
        "{\n"
        '  "research_topic": "string",\n'
        '  "interview_count": number,\n'
        '  "executive_summary": "3 sentences",\n'
        '  "methodology_notes": "sample size and caveats",\n'
        '  "confidence_level": "High|Medium|Low",\n'
        '  "key_themes_count": number,\n'
        '  "primary_theme": "the strongest theme in one sentence",\n'
        '  "themes": [\n'
        "    {\n"
        '      "title": "Short theme name",\n'
        '      "description": "1-2 sentence description",\n'
        '      "evidence_count": number,\n'
        '      "verbatim_quotes": ["quote 1", "quote 2"]\n'
        "    }\n"
        "  ],\n"
        '  "contradictions": [\n'
        '    "Describe a contradiction or tension across interviews, if any"\n'
        "  ],\n"
        '  "open_questions": [\n'
        '    "Questions the research left unanswered"\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- research_topic should be a clear topic statement, not a question\n"
        "- interview_count must match the number of transcripts provided\n"
        "- executive_summary must be exactly 3 sentences\n"
        "- methodology_notes should mention sample size and caveats\n"
        "- confidence_level must be exactly one of: High, Medium, Low\n"
        "- primary_theme should be specific not generic\n"
        "- themes should be 3-7 items, ranked by signal strength\n"
        "- verbatim_quotes should be exact phrases from transcripts\n"
        "- contradictions can be an empty list if none found\n"
        "- open_questions should be 2-4 items"
    )

    transcript_sections = []
    for i, t in enumerate(transcripts, start=1):
        section = (
            "=== TRANSCRIPT " + str(i) + ": " + t["filename"] + " ===\n\n"
            + t["text"] + "\n\n"
            "=== END OF TRANSCRIPT " + str(i) + " ==="
        )
        transcript_sections.append(section)

    combined_transcripts = "\n\n".join(transcript_sections)

    user_message = (
        "Here are " + str(len(transcripts)) + " user interview transcripts "
        'on the topic: "' + research_topic + '"\n\n'
        "Synthesize themes across all interviews. Find patterns that appear "
        "in multiple transcripts. Pull verbatim quotes as evidence. Surface "
        "contradictions between interviewees. Identify what the research left "
        "unanswered.\n\n"
        "TRANSCRIPTS:\n\n"
        + combined_transcripts + "\n\n"
        "Generate the synthesis now. Output JSON only."
    )

    return {
        "system": system_prompt,
        "user": user_message,
        "total_words": sum(t["word_count"] for t in transcripts)
    }


def call_claude_synthesizer(request):
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            temperature=0.2,
            system=request["system"],
            messages=[
                {"role": "user", "content": request["user"]}
            ]
        )
    except anthropic.APIConnectionError:
        raise Exception("Could not connect to Claude API. Check your internet.")
    except anthropic.AuthenticationError:
        raise Exception("Invalid Anthropic API key. Check your .env or Streamlit secrets.")
    except anthropic.RateLimitError:
        raise Exception("Rate limit hit. Wait a minute and try again.")
    except anthropic.APIError as e:
        raise Exception("Claude API error: " + str(e))

    response_text = response.content[0].text
    cleaned = response_text.strip()

    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise Exception(
            "Claude returned invalid JSON. Error: " + str(e)
            + "\n\nFirst 500 chars: " + cleaned[:500]
        )

    result["_meta"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "model": response.model,
        "stop_reason": response.stop_reason
    }

    return result

def format_synthesis_as_markdown(result, research_topic, transcripts):
    lines = []

    # Title
    lines.append("# Research Synthesis Report")
    lines.append("")
    lines.append("**Research topic:** " + research_topic)
    lines.append("**Date:** " + str(result.get("_meta", {}).get("model", "")).split("-")[0])
    lines.append("**Interviews:** " + str(result.get("interview_count", "—")))
    lines.append("**Confidence:** " + str(result.get("confidence_level", "—")))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Primary theme
    if "primary_theme" in result:
        lines.append("## Primary Theme")
        lines.append("")
        lines.append("> " + result["primary_theme"])
        lines.append("")

    # Executive summary
    if "executive_summary" in result:
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(result["executive_summary"])
        lines.append("")

    # Themes
    themes = result.get("themes", [])
    if themes:
        lines.append("## Themes")
        lines.append("")
        for i, theme in enumerate(themes, start=1):
            title = theme.get("title", "Theme " + str(i))
            description = theme.get("description", "")
            evidence_count = theme.get("evidence_count", 0)
            quotes = theme.get("verbatim_quotes", [])

            lines.append("### " + str(i) + ". " + title)
            lines.append("")
            lines.append("**Evidence:** " + str(evidence_count) + " interviews")
            lines.append("")
            if description:
                lines.append(description)
                lines.append("")
            if quotes:
                lines.append("**Verbatim quotes:**")
                lines.append("")
                for q in quotes:
                    lines.append("> *\"" + q + "\"*")
                    lines.append("")

    # Contradictions
    contradictions = result.get("contradictions", [])
    if contradictions:
        lines.append("## Contradictions Across Interviews")
        lines.append("")
        for c in contradictions:
            lines.append("- " + c)
        lines.append("")

    # Open questions
    open_questions = result.get("open_questions", [])
    if open_questions:
        lines.append("## Open Questions for Follow-Up")
        lines.append("")
        for q in open_questions:
            lines.append("- " + q)
        lines.append("")

    # Methodology
    if "methodology_notes" in result:
        lines.append("## Methodology Notes")
        lines.append("")
        lines.append(result["methodology_notes"])
        lines.append("")

    # Source transcripts
    lines.append("## Source Transcripts")
    lines.append("")
    for t in transcripts:
        lines.append("- " + t["filename"] + " (" + format(t["word_count"], ",") + " words)")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by PM Agent #2 of 24 · StreamMind · Built with Claude Sonnet 4.6*")

    return "\n".join(lines)


# Page config
st.set_page_config(
    page_title="PM Agent #2 — Research Synthesizer",
    page_icon="📋",
    layout="centered"
)

# Header
st.title("📋 Research Synthesizer")
st.markdown("*Drop 3-5 user interview transcripts. Get a structured synthesis report in seconds.*")

if not api_key:
    st.error("❌ Anthropic API key not configured. Contact the admin.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### About this agent")
    st.markdown(
        "This is **PM Agent #2** from StreamMind — a portfolio of 24 AI agents "
        "for streaming, product management, and pharma domains."
    )
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown(
        "1. Upload 3-5 interview transcripts\n"
        "2. Specify the research topic\n"
        "3. Click 'Synthesize'\n"
        "4. Get a structured synthesis report"
    )
    st.markdown("---")
    st.markdown("Built by [Shreya Patel](https://www.linkedin.com/) · [GitHub](https://github.com/)")

# Step 1
st.markdown("### Step 1: What's the research topic?")
research_topic = st.text_input(
    "Research topic",
    placeholder="e.g., Users navigating subscription downgrade flows",
    help="A clear topic statement helps Claude focus the synthesis"
)

# Step 2
st.markdown("### Step 2: Upload interview transcripts")
uploaded_files = st.file_uploader(
    "Drop 3-5 transcripts here (.txt, .pdf, or .docx)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
    help="Each file should be a single interview transcript"
)

parsed_transcripts = []
parsing_errors = []

if uploaded_files:
    st.success("✅ " + str(len(uploaded_files)) + " file(s) uploaded")

    with st.expander("View uploaded files", expanded=False):
        for f in uploaded_files:
            try:
                text = parse_uploaded_file(f)
                word_count = len(text.split())
                parsed_transcripts.append({
                    "filename": f.name,
                    "text": text,
                    "word_count": word_count
                })
                st.markdown("- ✅ **" + f.name + "** — " + format(word_count, ",") + " words extracted")
            except ValueError as e:
                parsing_errors.append(str(e))
                st.markdown("- ❌ **" + f.name + "** — " + str(e))

    if parsed_transcripts:
        with st.expander("Preview first transcript (first 500 chars)", expanded=False):
            preview = parsed_transcripts[0]["text"][:500]
            st.code(preview + "...", language=None)

    if parsing_errors:
        st.warning(
            "⚠️ " + str(len(parsing_errors)) + " file(s) could not be parsed."
        )

# Step 3
st.markdown("### Step 3: Run synthesis")
if st.button("🔍 Synthesize", type="primary", use_container_width=True):
    if not research_topic:
        st.warning("⚠️ Please enter a research topic")
    elif not parsed_transcripts:
        st.warning("⚠️ Please upload at least one transcript that can be parsed")
    elif len(parsed_transcripts) < 2:
        st.warning("⚠️ Upload at least 2 transcripts for cross-transcript synthesis")
    else:
        request = build_synthesis_request(research_topic, parsed_transcripts)

        with st.spinner(
            "🧠 Claude is synthesizing themes across "
            + str(len(parsed_transcripts)) + " transcripts ("
            + format(request["total_words"], ",") + " words)..."
        ):
            try:
                result = call_claude_synthesizer(request)
                st.session_state["synthesis_result"] = result
                st.session_state["synthesis_request"] = {
                    "research_topic": research_topic,
                    "total_words": request["total_words"]
                }
                st.session_state["parsed_transcripts_for_export"] = [
                    {"filename": t["filename"], "word_count": t["word_count"]}
                    for t in parsed_transcripts
                ]
            except Exception as e:
                st.error("❌ Synthesis failed: " + str(e))
                st.session_state.pop("synthesis_result", None)

# Display result if exists
# Display result if exists
if "synthesis_result" in st.session_state:
    result = st.session_state["synthesis_result"]

    st.markdown("---")

    # Action bar — download + clear
    col_dl, col_clear = st.columns([3, 1])
    with col_dl:
        markdown_export = format_synthesis_as_markdown(
            result,
            st.session_state.get("synthesis_request", {}).get("research_topic", "—"),
            st.session_state.get("parsed_transcripts_for_export", [])
        )
        filename_safe_topic = "".join(
            c for c in research_topic if c.isalnum() or c in (" ", "-", "_")
        ).strip().replace(" ", "_")[:40] or "research_synthesis"

        st.download_button(
            label="⬇️ Download as Markdown",
            data=markdown_export,
            file_name=filename_safe_topic + "_synthesis.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col_clear:
        if st.button("🔄 New synthesis", use_container_width=True):
            st.session_state.pop("synthesis_result", None)
            st.session_state.pop("synthesis_request", None)
            st.session_state.pop("parsed_transcripts_for_export", None)
            st.rerun()

    st.markdown("## 📊 Synthesis Report")

    # Top metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Interviews", result.get("interview_count", "—"))
    with col2:
        st.metric("Themes", result.get("key_themes_count", "—"))
    with col3:
        st.metric("Confidence", result.get("confidence_level", "—"))

    # Primary theme — the headline insight
    if "primary_theme" in result:
        st.markdown("### 🎯 Primary theme")
        st.info(result["primary_theme"])

    # Executive summary
    if "executive_summary" in result:
        st.markdown("### Executive summary")
        st.markdown(result["executive_summary"])

    # Themes section with verbatim quotes
    themes = result.get("themes", [])
    if themes:
        st.markdown("### 🔍 Themes")
        st.caption(
            "Ranked by signal strength across "
            + str(result.get("interview_count", "—"))
            + " interviews"
        )

        for i, theme in enumerate(themes, start=1):
            title = theme.get("title", "Theme " + str(i))
            description = theme.get("description", "")
            evidence_count = theme.get("evidence_count", 0)
            quotes = theme.get("verbatim_quotes", [])

            st.markdown(
                "#### " + str(i) + ". " + title
                + "  &nbsp;`" + str(evidence_count) + " interviews`"
            )

            if description:
                st.markdown(description)

            if quotes:
                with st.expander(
                    "💬 Verbatim quotes (" + str(len(quotes)) + ")",
                    expanded=(i == 1)
                ):
                    for q in quotes:
                        st.markdown("> *\"" + q + "\"*")
                        st.markdown("")

    # Contradictions
    contradictions = result.get("contradictions", [])
    if contradictions:
        st.markdown("### ⚡ Contradictions across interviews")
        st.caption("Tensions and disagreements worth investigating further")
        for c in contradictions:
            st.warning(c)

    # Open questions
    open_questions = result.get("open_questions", [])
    if open_questions:
        st.markdown("### ❓ Open questions for follow-up")
        st.caption("The research raised these but did not answer them")
        for q in open_questions:
            st.markdown("- " + q)

    # Methodology
    if "methodology_notes" in result:
        with st.expander("📋 Methodology notes", expanded=False):
            st.markdown(result["methodology_notes"])

    # Token usage caption
    if "_meta" in result:
        meta = result["_meta"]
        st.markdown("---")
        st.caption(
            "Model: " + str(meta["model"])
            + " · Input tokens: " + format(meta["input_tokens"], ",")
            + " · Output tokens: " + format(meta["output_tokens"], ",")
            + " · Stop reason: " + str(meta["stop_reason"])
        )

    # Raw JSON
    with st.expander("🔧 Raw JSON response (developer view)", expanded=False):
        st.json(result)
# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "PM Agent #2 of 24 · StreamMind · Built with Claude Sonnet 4.6"
    "</div>",
    unsafe_allow_html=True
)