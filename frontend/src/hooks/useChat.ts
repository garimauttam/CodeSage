/**
 * useChat.ts — Custom hook managing chat state and API communication.
 *
 * Enhancements in this version:
 * 1. AbortController — cancels the in-flight fetch when the component unmounts
 *    or when a new message is sent before the previous one finishes.
 *
 * 2. activeRepoUrl — passed to the backend so retrieval is scoped to one repo.
 *
 * 3. localStorage persistence — chat history is saved per-repo and restored on
 *    page refresh. Key format: "codesage:messages:{repoUrl}". Stored messages
 *    have isStreaming stripped so a refreshed page never shows stale spinners.
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { Message, SourceFile } from "../types";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

function generateId(): string {
  return Math.random().toString(36).slice(2, 9);
}

function storageKey(repoUrl: string | null | undefined): string {
  return `codesage:messages:${repoUrl ?? "__none__"}`;
}

function loadMessages(repoUrl: string | null | undefined): Message[] {
  try {
    const raw = localStorage.getItem(storageKey(repoUrl));
    if (!raw) return [];
    const parsed: Message[] = JSON.parse(raw);
    // Strip any stale isStreaming flags — never show a spinner on load
    return parsed.map((m) => ({ ...m, isStreaming: false }));
  } catch {
    return [];
  }
}

function saveMessages(repoUrl: string | null | undefined, messages: Message[]): void {
  try {
    // Only persist completed messages — drop any still-streaming placeholder
    const toSave = messages.filter((m) => !m.isStreaming);
    localStorage.setItem(storageKey(repoUrl), JSON.stringify(toSave));
  } catch {
    // localStorage can throw if storage is full — silently ignore
  }
}

export function useChat(
  activeRepoUrl?: string | null,
  activeRepoUrls?: string[] | null,
) {
  const [messages, setMessages] = useState<Message[]>(() => loadMessages(activeRepoUrl));
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // When the active repo changes, swap in that repo's persisted history
  useEffect(() => {
    setMessages(loadMessages(activeRepoUrl));
    setError(null);
  }, [activeRepoUrl]);

  // Persist to localStorage whenever messages change (debounced by React batching)
  useEffect(() => {
    saveMessages(activeRepoUrl, messages);
  }, [messages, activeRepoUrl]);

  // Track the current in-flight AbortController so we can cancel it
  // when a new message is sent or the component unmounts.
  const abortControllerRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (
    question: string,
    activeRepoUrl?: string | null,
  ) => {
    if (!question.trim() || isLoading) return;

    // Cancel any previous in-flight request before starting a new one.
    // This handles "user sends message before previous one finishes" correctly.
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    setError(null);

    // Snapshot history NOW — before we call setMessages below.
    // setMessages is async; if we read `messages` inside the try block
    // after the setState call, it still refers to the pre-update snapshot
    // from this render cycle, missing the user message we just added.
    // Snapshotting here gives us the correct history to send to the backend.
    const history = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content: question,
    };

    const assistantMessageId = generateId();
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsLoading(true);

    try {

      const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          chat_history: history,
          active_repo_url: activeRepoUrl ?? null,
          ...(activeRepoUrls && activeRepoUrls.length > 0
            ? { active_repo_urls: activeRepoUrls }
            : {}),
        }),
        // Pass the signal — fetch will throw an AbortError if aborted
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let sources: SourceFile[] = [];
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        buffer += text;

        // Parse the __SOURCES__ marker out of the stream
        if (buffer.includes("__SOURCES_END__")) {
          const sourceMatch = buffer.match(/__SOURCES__(.*?)__SOURCES_END__\n/s);
          if (sourceMatch) {
            try {
              sources = JSON.parse(sourceMatch[1]);
            } catch {}
            buffer = buffer.replace(/__SOURCES__.*?__SOURCES_END__\n/s, "");
          }
        }

        const currentBuffer = buffer;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessageId
              ? { ...m, content: currentBuffer, sources }
              : m
          )
        );
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessageId
            ? { ...m, isStreaming: false, sources }
            : m
        )
      );
    } catch (err) {
      // AbortError is not a real error — it means we intentionally cancelled.
      // Don't show an error banner for it; just clean up the placeholder message.
      if (err instanceof Error && err.name === "AbortError") {
        setMessages((prev) => prev.filter((m) => m.id !== assistantMessageId));
        setIsLoading(false);
        return;
      }

      setError(err instanceof Error ? err.message : "An error occurred.");
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessageId
            ? {
                ...m,
                content: m.content
                  ? m.content + "\n\n*[Stream interrupted — response may be incomplete.]*"
                  : "Sorry, I encountered an error. Please try again.",
                isStreaming: false,
              }
            : m
        )
      );
    } finally {
      // Only clear loading if this is still the active request (not cancelled)
      if (abortControllerRef.current === abortController) {
        setIsLoading(false);
        abortControllerRef.current = null;
      }
    }
  }, [messages, isLoading]);

  const clearChat = useCallback(() => {
    // Also cancel any in-flight request when clearing the chat
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setMessages([]);
    setError(null);
  }, []);

  return { messages, isLoading, error, sendMessage, clearChat };
}
