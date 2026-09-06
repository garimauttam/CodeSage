/**
 * useMultiReview.ts — State and streaming logic for multi-file code review.
 *
 * The multi-review stream adds section delimiters on top of the STATUS protocol:
 *   __SECTION_START__filename__SECTION_END__   → start a new per-file card
 *   __STATUS__...{json}__STATUS_END__          → progress update (same as single review)
 *   everything else                            → review markdown for the current section
 *
 * We maintain a `sections` array so the UI can render each file as a collapsible card.
 */

import { useState, useCallback } from "react";
import { IndexedFile } from "../types";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

export interface ReviewSection {
  fileName: string;
  content: string;
}

export interface AgentStep {
  tool: string;
  message: string;
}

export interface MultiReviewState {
  isReviewing: boolean;
  sections: ReviewSection[];    // one per file + optional summary
  agentSteps: AgentStep[];
  currentStep: string | null;
  error: string | null;
}

export function useMultiReview() {
  const [state, setState] = useState<MultiReviewState>({
    isReviewing: false,
    sections: [],
    agentSteps: [],
    currentStep: null,
    error: null,
  });

  const reviewFiles = useCallback(async (files: IndexedFile[]) => {
    setState({ isReviewing: true, sections: [], agentSteps: [], currentStep: "Starting multi-file review...", error: null });

    try {
      const response = await fetch(`${API_BASE}/api/v1/review/multi`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          files: files.map((f) => ({
            file_path: f.source,
            file_name: f.file_name,
            language: f.language,
          })),
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: `Server error ${response.status}` }));
        throw new Error(err.detail ?? `Server error ${response.status}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let currentSectionName = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // ── ERROR terminal marker ─────────────────────────────────────────────
        // Surfaced as an in-section error note so the rest of the multi-review
        // output remains visible (earlier sections are not lost).
        if (buffer.includes("__ERROR__") && buffer.includes("__ERROR_END__")) {
          const eStart = buffer.indexOf("__ERROR__");
          const eEnd   = buffer.indexOf("__ERROR_END__");
          if (eStart !== -1 && eEnd !== -1) {
            const errMsg = buffer.slice(eStart + 9, eEnd).trim();
            buffer = buffer.slice(eEnd + 13 + 1); // consume the marker
            const displayMsg = `\n\n> ⚠️ **Error:** ${errMsg || "Server error — response may be incomplete."}`;
            if (currentSectionName) {
              setState((prev) => ({
                ...prev,
                sections: prev.sections.map((s) =>
                  s.fileName === currentSectionName
                    ? { ...s, content: s.content + displayMsg }
                    : s
                ),
              }));
            }
          }
        }

        // Process all complete markers in the buffer
        let changed = true;
        while (changed) {
          changed = false;

          // ── Section start marker ───────────────────────────────────────────
          if (buffer.includes("__SECTION_START__") && buffer.includes("__SECTION_END__")) {
            const sStart = buffer.indexOf("__SECTION_START__");
            const sEnd = buffer.indexOf("__SECTION_END__");
            if (sStart !== -1 && sEnd !== -1 && sEnd > sStart) {
              // Flush any buffered text to the current section first
              const textBefore = buffer.slice(0, sStart);
              if (textBefore.trim() && currentSectionName) {
                setState((prev) => ({
                  ...prev,
                  sections: prev.sections.map((s) =>
                    s.fileName === currentSectionName
                      ? { ...s, content: s.content + textBefore }
                      : s
                  ),
                }));
              }

              const newSectionName = buffer.slice(sStart + 17, sEnd); // strip __SECTION_START__
              buffer = buffer.slice(sEnd + 15 + 1); // strip __SECTION_END__\n
              currentSectionName = newSectionName;

              setState((prev) => ({
                ...prev,
                sections: [...prev.sections, { fileName: newSectionName, content: "" }],
              }));
              changed = true;
              continue;
            }
          }

          // ── STATUS marker ──────────────────────────────────────────────────
          if (buffer.includes("__STATUS__") && buffer.includes("__STATUS_END__")) {
            const start = buffer.indexOf("__STATUS__");
            const end = buffer.indexOf("__STATUS_END__");
            if (start !== -1 && end !== -1) {
              // Flush text before this marker to current section
              const textBefore = buffer.slice(0, start);
              if (textBefore && currentSectionName) {
                setState((prev) => ({
                  ...prev,
                  sections: prev.sections.map((s) =>
                    s.fileName === currentSectionName
                      ? { ...s, content: s.content + textBefore }
                      : s
                  ),
                }));
              }

              const statusText = buffer.slice(start + 10, end);
              buffer = buffer.slice(end + 14 + 1);

              const jsonMatch = statusText.match(/(\{.*\})$/);
              const message = jsonMatch
                ? statusText.slice(0, statusText.lastIndexOf(jsonMatch[0])).trim()
                : statusText.trim();
              let meta: { step?: string; tool?: string } = {};
              if (jsonMatch) {
                try { meta = JSON.parse(jsonMatch[1]); } catch {}
              }

              setState((prev) => ({
                ...prev,
                currentStep: message,
                agentSteps: meta.tool
                  ? [...prev.agentSteps, { tool: meta.tool, message }]
                  : prev.agentSteps,
              }));
              changed = true;
            }
          }
        }

        // Flush plain text that's not a partial marker
        if (
          !buffer.includes("__SECTION_START__") &&
          !buffer.includes("__STATUS__") &&
          currentSectionName
        ) {
          const partialMatch = buffer.match(/_{1,2}(?:S(?:E(?:C(?:T(?:I(?:O(?:N)?)?)?)?)?|T(?:A(?:T(?:U(?:S)?)?)?)?)?)?$/);
          const splitIdx = partialMatch ? (partialMatch.index ?? buffer.length) : buffer.length;
          const text = buffer.slice(0, splitIdx);
          buffer = buffer.slice(splitIdx);
          if (text) {
            setState((prev) => ({
              ...prev,
              sections: prev.sections.map((s) =>
                s.fileName === currentSectionName
                  ? { ...s, content: s.content + text }
                  : s
              ),
            }));
          }
        }
      }

      setState((prev) => ({ ...prev, isReviewing: false, currentStep: null }));
    } catch (err) {
      setState((prev) => ({
        ...prev,
        isReviewing: false,
        error: err instanceof Error ? err.message : "Review failed.",
      }));
    }
  }, []);

  const reset = useCallback(() => {
    setState({ isReviewing: false, sections: [], agentSteps: [], currentStep: null, error: null });
  }, []);

  return { ...state, reviewFiles, reset };
}
