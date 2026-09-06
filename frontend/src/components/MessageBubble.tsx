/**
 * MessageBubble.tsx — Renders a single chat message.
 *
 * WHY react-markdown + react-syntax-highlighter?
 * The LLM returns markdown: **bold**, `code`, ```python blocks```.
 * react-markdown parses this into proper HTML.
 * react-syntax-highlighter adds color-coded syntax highlighting to code blocks.
 * Without these, the user would see raw markdown text, which looks terrible.
 *
 * WHY A BLINKING CURSOR?
 * When isStreaming=true, we append a blinking ▋ to the message.
 * This gives clear visual feedback that the answer is still being generated.
 * Without it, the UI looks frozen between tokens.
 */

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { User, Bot, FileCode } from "lucide-react";
import { Message } from "../types";

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
          isUser ? "bg-blue-600" : "bg-purple-700"
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-white" />
        )}
      </div>

      <div className={`flex flex-col gap-2 max-w-[80%] ${isUser ? "items-end" : "items-start"}`}>
        {/* Message bubble */}
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? "bg-blue-600 text-white rounded-tr-sm"
              : "bg-gray-800 text-gray-100 rounded-tl-sm"
          }`}
        >
          {isUser ? (
            // User messages are plain text — no markdown needed
            <p>{message.content}</p>
          ) : (
            // Assistant messages are markdown — render properly
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                // Custom renderer for code blocks — adds syntax highlighting
                code({ className, children }: any) {
                    const match = /language-(\w+)/.exec(className || "");
                    const isBlock = Boolean(match);
  
                    return isBlock && match ? (
                    <SyntaxHighlighter
                      style={vscDarkPlus}
                      language={match[1]}
                      PreTag="div"
                      className="rounded-lg text-xs my-2"
                    >
                      {String(children).replace(/\n$/, "")}
                    </SyntaxHighlighter>
                  ) : (
                    // Inline code (backtick)
                    <code className="bg-gray-700 text-purple-300 px-1.5 py-0.5 rounded text-xs font-mono">
                      {children}
                    </code>
                  );
                },
                // Make links open in new tab
                a({ children, href }) {
                  return (
                    <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-400 underline">
                      {children}
                    </a>
                  );
                },
              }}
            >
              {message.content + (message.isStreaming ? " ▋" : "")}
            </ReactMarkdown>
          )}
        </div>

        {/* Sources panel — only shown when the assistant has sources and is done streaming */}
        {!isUser && message.sources && message.sources.length > 0 && !message.isStreaming && (
          <div className="flex flex-wrap gap-1.5">
            {message.sources.map((src, i) => (
              <span
                key={i}
                title={src.source}
                className="flex items-center gap-1 bg-gray-900 border border-gray-700 text-gray-400 text-xs px-2 py-0.5 rounded-full"
              >
                <FileCode className="w-3 h-3 text-purple-400" />
                {src.file_name}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
