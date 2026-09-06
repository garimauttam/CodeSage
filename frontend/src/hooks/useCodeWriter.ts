/**
 * useCodeWriter.ts — State and streaming logic for the code writing agent.
 *
 * Identical stream-parsing approach to useReview.ts:
 * __STATUS__...{json}__STATUS_END__ markers → progress trace
 * everything else                           → generated code output
 */

import { useState, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

export interface WriterStep {
  step: string;
  message: string;
}

export interface CodeWriterState {
  isGenerating: boolean;
  output: string;           // streamed code + explanation
  steps: WriterStep[];      // progress trace
  currentStep: string | null;
  error: string | null;
}

export function useCodeWriter() {
  const [state, setState] = useState<CodeWriterState>({
    isGenerating: false,
    output: "",
    steps: [],
    currentStep: null,
    error: null,
  });

  const generate = useCallback(async (
    prompt: string,
    language: string,
    fileName: string,
    contextSources: string[],
    mode: "generate" | "edit" | "tests" = "generate",
  ) => {
    setState({ isGenerating: true, output: "", steps: [], currentStep: "Starting...", error: null });

    try {
      const response = await fetch(`${API_BASE}/api/v1/write/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          language,
          file_name: fileName,
          context_sources: contextSources,
          mode,
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: `Server error ${response.status}` }));
        throw new Error(err.detail ?? `Server error ${response.status}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // ── ERROR terminal marker ─────────────────────────────────────────────
        if (buffer.includes("__ERROR__") && buffer.includes("__ERROR_END__")) {
          const eStart = buffer.indexOf("__ERROR__");
          const eEnd   = buffer.indexOf("__ERROR_END__");
          if (eStart !== -1 && eEnd !== -1) {
            const errMsg = buffer.slice(eStart + 9, eEnd).trim();
            setState((prev) => ({
              ...prev,
              isGenerating: false,
              currentStep: null,
              error: errMsg || "Generation was interrupted by a server error.",
            }));
            return;
          }
        }

        // Drain STATUS markers
        while (buffer.includes("__STATUS_END__")) {
          const start = buffer.indexOf("__STATUS__");
          const end = buffer.indexOf("__STATUS_END__");
          if (start === -1 || end === -1) break;

          const statusText = buffer.slice(start + 10, end);
          buffer = buffer.slice(end + 14 + 1); // +1 for trailing \n

          const jsonMatch = statusText.match(/(\{.*\})$/);
          const message = jsonMatch
            ? statusText.slice(0, statusText.lastIndexOf(jsonMatch[0])).trim()
            : statusText.trim();
          let meta: { step?: string } = {};
          if (jsonMatch) {
            try { meta = JSON.parse(jsonMatch[1]); } catch {}
          }

          setState((prev) => ({
            ...prev,
            currentStep: message,
            steps: meta.step
              ? [...prev.steps, { step: meta.step, message }]
              : prev.steps,
          }));
        }

        // Flush non-STATUS buffer content to output
        if (!buffer.includes("__STATUS__")) {
          const partialMatch = buffer.match(/_{1,2}(?:S(?:T(?:A(?:T(?:U(?:S)?)?)?)?)?)?$/);
          if (partialMatch) {
            const splitIdx = partialMatch.index ?? buffer.length;
            const text = buffer.slice(0, splitIdx);
            buffer = buffer.slice(splitIdx);
            if (text) setState((prev) => ({ ...prev, output: prev.output + text }));
          } else {
            const text = buffer;
            buffer = "";
            if (text) setState((prev) => ({ ...prev, output: prev.output + text }));
          }
        }
      }

      setState((prev) => ({ ...prev, isGenerating: false, currentStep: null }));
    } catch (err) {
      setState((prev) => ({
        ...prev,
        isGenerating: false,
        error: err instanceof Error ? err.message : "Generation failed.",
      }));
    }
  }, []);

  const reset = useCallback(() => {
    setState({ isGenerating: false, output: "", steps: [], currentStep: null, error: null });
  }, []);

  return { ...state, generate, reset };
}
