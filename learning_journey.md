# Learning Journey: Building a Kanglish Voice Agent

This document captures the technical challenges, experiments, and pivotal decisions made during the development of the Voice Agent, specifically focusing on the transition from general-purpose APIs to specialized Indic models.

## Phase 1: The Generalist Approach (ElevenLabs + Ambari)

**Initial Architecture:**
- **STT**: ElevenLabs (English)
- **TTS**: ElevenLabs (English)
- **Translation**: Ambari-7B (Fine-tuned for Kanglish <-> English)

**Observations:**
Initially, we used ElevenLabs for both Speech-to-Text (STT) and Text-to-Speech (TTS) with English as the primary language. This setup worked reliably for standard English interactions.

## Phase 2: The Kanglish Integration Challenge

**The Goal:**
Enable "Kanglish" (Kannada + English mix) support by integrating the Ambari-7B model as a translation layer between the user and the agent.

**Challenge 1: STT Language Detection**
*   **Attempt:** relying on ElevenLabs' auto-detect feature.
*   **Result:** **Failure**. The auto-detect mechanism struggled significantly with Kanglish, often misinterpreting Kannada phonemes as gibberish English or failing to transcribe them entirely.

**Challenge 2: Forcing Language Constraints**
*   **Attempt:** We explicitly set the ElevenLabs STT language parameter to `kan` (Kannada).
*   **Result:** **Suboptimal**. While it improved detection slightly, the transcription quality was still "not up to the point." It missed nuances and often outputted incorrect transliterations that confused the translation layer.

## Phase 3: Root Cause Analysis

We needed to isolate whether the issue lay with the Transcription (ElevenLabs) or the Translation (Ambari).

1.  **Evaluated Ambari:** We tested the Ambari-7B model independently on text inputs.
    *   **Result:** The model demonstrated approximately **80% accuracy** in translating Kanglish to English and vice-versa.
    *   **Conclusion:** Ambari was *not* the primary bottleneck.

2.  **Re-evaluated STT:** We analyzed the raw transcripts coming from ElevenLabs.
    *   **Result:** The STT was consistently failing to capture the correct phonetic structure of the spoken Kannada/Kanglish. If the input garbage-in, the translation was inevitably garbage-out.

## Phase 4: The Pivot to Specialized Indic Models

**Decision:**
To achieve high-quality voice interactions in Indian languages, generic multi-lingual models were insufficient. We decided to migrate the entire voice pipeline to specialized open-source models developed specifically for Indic languages (mainly by AI4Bharat).

**New Architecture:**
1.  **STT**: **IndicConformer** (or similar Indic-specific ASR). Chosen for its robust handling of Indian native languages and dialects.
2.  **Translation**: **IndicTrans2 (1B)**.
    *   Replaced Ambari-7B.
    *   **Reasoning:** The 1B parameter models (En-Indic and Indic-En) are lighter, faster, and purpose-built for high-fidelity translation, offering a better trade-off than a generic 7B LLM for this specific task.
3.  **TTS**: **IndicParler**. A specialized TTS model capable of generating natural-sounding speech in Indian languages, replacing the generic ElevenLabs voices which lacked native prosody.

**Outcome:**
This migration aims to solve the fundamental data loss occurring at the STT layer while optimizing the translation latency and accuracy.
