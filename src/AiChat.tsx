import { createSignal, createEffect, onCleanup, Show } from "solid-js";
import { invoke, Channel } from "@tauri-apps/api/core";
import AIConversation, { Message } from "./AIConversation";
import { SessionDetails } from "./App";

interface AiChatProps {
  session: SessionDetails;
}

// Define the structured event we expect from Rust
interface AgentEvent {
  type: "Status" | "Chunk";
  data: string;
}

export default function AiChat(props: AiChatProps) {
  const [messages, setMessages] = createSignal<Message[]>([
    {
      role: "system",
      content: "You are a helpful coding assistant.",
    },
  ]);

  const [input, setInput] = createSignal("");
  const [isGenerating, setIsGenerating] = createSignal(false);
  const [activeStatus, setActiveStatus] = createSignal<string | null>(null);
  const [statusElapsedMs, setStatusElapsedMs] = createSignal(0);

  let chatContainerRef: HTMLDivElement | undefined;
  let scrollPending = false;
  let statusStartedAt = 0;
  let statusTimer: number | undefined;

  const formatElapsed = (ms: number) => {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return minutes > 0
      ? `${minutes}:${seconds.toString().padStart(2, "0")}`
      : `${seconds}s`;
  };

  const updateStatus = (status: string | null) => {
    setActiveStatus(status);
    statusStartedAt = status ? Date.now() : 0;
    setStatusElapsedMs(0);
  };

  const scheduleScroll = () => {
    if (scrollPending) return;
    scrollPending = true;
    requestAnimationFrame(() => {
      if (chatContainerRef) {
        chatContainerRef.scrollTop = chatContainerRef.scrollHeight;
      }
      scrollPending = false;
    });
  };

  createEffect(() => {
    messages();
    activeStatus(); // Also scroll if the status pill changes
    scheduleScroll();
  });

  onCleanup(() => {
    scrollPending = false;
    if (statusTimer) window.clearInterval(statusTimer);
  });

  statusTimer = window.setInterval(() => {
    if (!activeStatus() || statusStartedAt === 0) {
      setStatusElapsedMs(0);
      return;
    }
    setStatusElapsedMs(Date.now() - statusStartedAt);
  }, 1000);

  async function handleSubmit(e?: Event) {
    e?.preventDefault();
    const currentInput = input().trim();
    if (!currentInput || isGenerating()) return;

    // 1. Instantly append user message to UI
    setMessages((prev) => [...prev, { role: "user", content: currentInput }]);
    setInput("");
    setIsGenerating(true);
    updateStatus("Getting the request ready..."); // Initial status before stream begins

    // 2. Create empty assistant bubble
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      // 3. Create a Tauri IPC Channel expecting our typed event
      const onChunk = new Channel<AgentEvent>();

      onChunk.onmessage = (event: AgentEvent) => {
        if (event.type === "Status") {
          // Update the status pill for tool execution
          updateStatus(event.data);
        } else if (event.type === "Chunk") {
          // As soon as real text arrives, clear the status pill
          if (activeStatus()) updateStatus(null);

          setMessages((prev) => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
            if (lastIndex >= 0 && updated[lastIndex].role === "assistant") {
              updated[lastIndex] = {
                ...updated[lastIndex],
                content: updated[lastIndex].content + event.data,
              };
            }
            return updated;
          });
        }
      };

      // 4. Invoke the Rust brain
      await invoke("prompt_agent_v2", {
        sessionId: props.session.id,
        question: currentInput,
        onChunk: onChunk,
      });
    } catch (error) {
      console.error("Inference Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "**Error:** Failed to execute request via local engine.",
        },
      ]);
    } finally {
      setIsGenerating(false);
      updateStatus(null);
    }
  }

  return (
    <div class="flex flex-col h-full w-full mx-auto bg-gray-950">
      <div class="flex items-center justify-between p-4 border-b border-gray-800 bg-gray-950/95 sticky top-0 z-10">
        <h2 class="text-sm font-semibold text-gray-200">
          {props.session.title}
        </h2>
      </div>

      <div
        ref={chatContainerRef}
        class="flex-1 overflow-y-auto py-4 space-y-6 scroll-smooth"
      >
        <AIConversation messages={messages()} />

        {/* Dynamic Status Pill for Tool Execution */}
        <Show when={activeStatus()}>
          <div class="flex justify-start px-4">
            <div class="flex max-w-[min(48rem,100%)] items-start gap-2 rounded-2xl border border-gray-700 bg-gray-800/80 px-3 py-2 text-xs leading-relaxed text-gray-400">
              <span class="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400 animate-ping"></span>
              <span class="min-w-0 whitespace-normal break-words">
                {activeStatus()}
              </span>
              <span class="ml-1 shrink-0 tabular-nums text-gray-500">
                {formatElapsed(statusElapsedMs())}
              </span>
            </div>
          </div>
        </Show>

        {/* Default pulsing dots before stream begins (only if no status pill is showing) */}
        <Show
          when={
            isGenerating() &&
            !activeStatus() &&
            messages()[messages().length - 1]?.content === ""
          }
        >
          <div class="flex justify-start px-4">
            <div class="bg-gray-800 border border-gray-700 rounded-2xl rounded-bl-sm px-5 py-3 flex items-center gap-1">
              <span class="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce"></span>
              <span
                class="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce"
                style="animation-delay:150ms"
              ></span>
              <span
                class="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce"
                style="animation-delay:300ms"
              ></span>
            </div>
          </div>
        </Show>
      </div>

      <div class="p-4 bg-gray-950 border-t border-gray-800">
        <form onSubmit={handleSubmit} class="relative max-w-3xl mx-auto">
          <textarea
            value={input()}
            onInput={(e) => setInput(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
            placeholder="Ask about your codebase..."
            class="w-full bg-gray-900 border border-gray-700 text-gray-100 rounded-xl px-4 py-3 pr-14 focus:outline-none focus:border-gray-500 resize-none shadow-inner"
            rows="1"
            disabled={isGenerating()}
          />

          <button
            type="submit"
            disabled={!input().trim() || isGenerating()}
            class="absolute right-2 bottom-2 p-2 text-gray-400 hover:text-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              class="w-5 h-5"
            >
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </form>
        <p class="text-center text-xs text-gray-600 mt-3">
          Shift + Enter for newline.
        </p>
      </div>
    </div>
  );
}
