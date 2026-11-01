import { createSignal, createEffect, onMount, onCleanup, Show } from "solid-js";
import { invoke, Channel } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import AIConversation, { Message, SourceRef } from "./AIConversation";
import { SessionDetails } from "./App";
import { NotebookEntryView } from "./types/notebook";

export interface PerformanceMetrics {
  current_tps: number;
  average_tps: number;
  total_tokens: number;
  is_cpu_bound: boolean;
  offload_info?: string | null;
}

interface AiChatProps {
  session: SessionDetails;
  indexingStatus?: "idle" | "indexing" | "ready";
  onNotebookUpdate?: (entries: NotebookEntryView[]) => void;
}

// Define the structured event we expect from Rust
interface AgentEvent {
  type: "Status" | "Chunk" | "Sources" | "Notebook" | "Diff" | "Performance";
  data: string | SourceRef[] | NotebookEntryView[] | PerformanceMetrics;
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
  const [autoApply, setAutoApply] = createSignal(
    props.session.auto_apply_edits !== undefined
      ? props.session.auto_apply_edits
      : true,
  );
  const [perfMetrics, setPerfMetrics] = createSignal<PerformanceMetrics | null>(null);
  const [errorMessage, setErrorMessage] = createSignal<string | null>(null);

  onMount(async () => {
    try {
      const initial = await invoke<PerformanceMetrics>("get_ai_performance");
      if (initial) setPerfMetrics(initial);
    } catch (e) {
      console.warn("Could not fetch initial AI performance:", e);
    }

    const unlisten = await listen<PerformanceMetrics>("ai-performance", (event) => {
      setPerfMetrics(event.payload);
    });

    onCleanup(() => {
      unlisten();
    });
  });

  const toggleAutoApply = async () => {
    const nextVal = !autoApply();
    setAutoApply(nextVal);
    try {
      await invoke("set_conversation_auto_apply", {
        sessionId: props.session.id,
        autoApply: nextVal,
      });
    } catch (e) {
      console.error("Failed to update auto apply mode:", e);
    }
  };

  let chatContainerRef: HTMLDivElement | undefined;
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

  // Load conversation history whenever session changes
  createEffect(async () => {
    const sessionId = props.session.id;
    if (!sessionId) return;
    setErrorMessage(null);
    try {
      const details = await invoke<any>("get_session_details", { sessionId });
      if (details && details.conversation && details.conversation.auto_apply_edits !== undefined) {
        setAutoApply(details.conversation.auto_apply_edits);
      }
      if (details && details.messages && details.messages.length > 0) {
        // Find the index of the last assistant message with a valid (non-empty) response
        let lastValidAssistantIndex = -1;
        for (let i = 0; i < details.messages.length; i++) {
          const m = details.messages[i];
          if (
            m.role &&
            m.role.toLowerCase() === "assistant" &&
            typeof m.content === "string" &&
            m.content.trim().length > 0
          ) {
            lastValidAssistantIndex = i;
          }
        }

        const validMessages =
          lastValidAssistantIndex >= 0
            ? details.messages.slice(0, lastValidAssistantIndex + 1)
            : [];

        setMessages([
          { role: "system", content: "You are a helpful coding assistant." },
          ...validMessages.map((m: any) => ({
            role: m.role.toLowerCase() as "user" | "assistant" | "system",
            content: m.content,
          })),
        ]);
      } else {
        setMessages([
          { role: "system", content: "You are a helpful coding assistant." },
        ]);
      }
    } catch (e) {
      console.error("Failed to load conversation history:", e);
    }
  });

  const updateStatus = (status: string | null) => {
    setActiveStatus(status);
    statusStartedAt = status ? Date.now() : 0;
    setStatusElapsedMs(0);
  };

  const scheduleScroll = () => {
    queueMicrotask(() => {
      if (chatContainerRef) {
        chatContainerRef.scrollTop = chatContainerRef.scrollHeight;
      }
    });
  };

  createEffect(() => {
    messages();
    activeStatus(); // Also scroll if the status pill changes
    scheduleScroll();
  });

  onCleanup(() => {
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
    setErrorMessage(null);

    // 1. Instantly append user message to UI
    setMessages((prev) => [...prev, { role: "user", content: currentInput }]);
    setInput("");
    setIsGenerating(true);
    updateStatus("Preparing the request"); // Initial status before stream begins

    // 2. Create empty assistant bubble
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      // 3. Create a Tauri IPC Channel expecting our typed event
      const onChunk = new Channel<AgentEvent>();

      onChunk.onmessage = (event: AgentEvent) => {
        if (event.type === "Status") {
          // Update the status pill for tool execution
          updateStatus(event.data as string);
        } else if (event.type === "Chunk") {
          // As soon as real text arrives, clear the status pill
          if (activeStatus()) updateStatus(null);

          setMessages((prev) => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
            if (lastIndex >= 0 && updated[lastIndex].role === "assistant") {
              updated[lastIndex] = {
                ...updated[lastIndex],
                content: updated[lastIndex].content + (event.data as string),
              };
            }
            return updated;
          });
        } else if (event.type === "Sources") {
          setMessages((prev) => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
            if (lastIndex >= 0 && updated[lastIndex].role === "assistant") {
              updated[lastIndex] = {
                ...updated[lastIndex],
                sources: event.data as SourceRef[],
              };
            }
            return updated;
          });
        } else if (event.type === "Diff") {
          setMessages((prev) => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
            if (lastIndex >= 0 && updated[lastIndex].role === "assistant") {
              updated[lastIndex] = {
                ...updated[lastIndex],
                diff: event.data as string,
                applied: autoApply(),
              };
            }
            return updated;
          });
        } else if (event.type === "Notebook") {
          props.onNotebookUpdate?.(event.data as NotebookEntryView[]);
        } else if (event.type === "Performance") {
          setPerfMetrics(event.data as PerformanceMetrics);
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
      setInput(currentInput);
      setErrorMessage("Failed to execute request via local engine. The incomplete input was pruned.");
      setMessages((prev) => {
        let lastValidAssistantIndex = -1;
        for (let i = 0; i < prev.length; i++) {
          if (
            prev[i].role === "assistant" &&
            typeof prev[i].content === "string" &&
            prev[i].content.trim().length > 0
          ) {
            lastValidAssistantIndex = i;
          }
        }
        if (lastValidAssistantIndex >= 0) {
          return prev.slice(0, lastValidAssistantIndex + 1);
        }
        return [{ role: "system", content: "You are a helpful coding assistant." }];
      });
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
        <div class="flex items-center gap-2">
          <Show when={perfMetrics() && perfMetrics()!.average_tps > 0}>
            <div
              class="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border text-[11px] font-mono transition-colors"
              classList={{
                "border-amber-700/80 bg-amber-950/60 text-amber-300": perfMetrics()!.is_cpu_bound,
                "border-gray-800 bg-gray-900/80 text-gray-400": !perfMetrics()!.is_cpu_bound,
              }}
              title={
                perfMetrics()!.is_cpu_bound
                  ? `Inference is CPU-bound! Speed: ${perfMetrics()!.average_tps.toFixed(1)} t/s (${perfMetrics()!.offload_info || "CPU"})`
                  : `Average generation speed: ${perfMetrics()!.average_tps.toFixed(1)} t/s (Current: ${perfMetrics()!.current_tps.toFixed(1)} t/s) • ${perfMetrics()!.offload_info || "Metal GPU"}`
              }
            >
              <span
                class="w-1.5 h-1.5 rounded-full"
                classList={{
                  "bg-amber-400 animate-pulse": perfMetrics()!.is_cpu_bound,
                  "bg-emerald-400": !perfMetrics()!.is_cpu_bound,
                }}
              ></span>
              <span>{perfMetrics()!.average_tps.toFixed(1)} t/s</span>
              <Show when={perfMetrics()!.is_cpu_bound}>
                <span class="text-[9px] font-sans font-bold text-amber-400 uppercase tracking-wider ml-0.5">CPU</span>
              </Show>
            </div>
          </Show>
        </div>
      </div>

      <div
        ref={chatContainerRef}
        class="flex-1 overflow-y-auto py-4 space-y-6 scroll-smooth"
      >
        {/* Prominent warning banner when operations are CPU-bound */}
        <Show when={perfMetrics()?.is_cpu_bound}>
          <div class="mx-4 my-2 p-3 rounded-xl border border-amber-600/40 bg-amber-950/40 text-amber-200 shadow-sm flex items-start justify-between gap-3 text-xs animate-pulse">
            <div class="flex items-start gap-2.5">
              <span class="text-base leading-none mt-0.5">⚠️</span>
              <div>
                <div class="font-semibold text-amber-100 flex items-center gap-1.5">
                  <span>CPU-Bound Inference Detected</span>
                  <span class="font-mono text-[11px] px-1.5 py-0.2 rounded bg-amber-900/60 text-amber-300 border border-amber-700/50">
                    {perfMetrics()?.average_tps ? `${perfMetrics()!.average_tps.toFixed(1)} t/s` : "Low throughput"}
                  </span>
                </div>
                <p class="text-amber-300/90 mt-1 leading-relaxed">
                  The local engine appears to be executing on CPU cores rather than Metal GPU acceleration. Inference will be noticeably slower (~4–8x slower).
                </p>
                <Show when={perfMetrics()?.offload_info}>
                  <div class="mt-1 text-[11px] font-mono text-amber-400/90">
                    Engine Status: {perfMetrics()?.offload_info}
                  </div>
                </Show>
              </div>
            </div>
          </div>
        </Show>

        <Show when={errorMessage()}>
          <div class="mx-4 my-2 p-3 rounded-xl border border-red-800/60 bg-red-950/40 text-red-200 text-xs flex items-center justify-between gap-2 shadow-sm">
            <div class="flex items-center gap-2">
              <span class="text-base leading-none">⚠️</span>
              <span>{errorMessage()}</span>
            </div>
            <button
              type="button"
              onClick={() => setErrorMessage(null)}
              class="text-red-400 hover:text-red-200 text-xs font-semibold px-2 py-0.5 rounded border border-red-800/40 hover:bg-red-900/30 transition-colors"
            >
              Dismiss
            </button>
          </div>
        </Show>

        <AIConversation messages={messages()} />

        {/* Dynamic Status Pill for Tool Execution */}
        <Show when={activeStatus()}>
          <div class="flex justify-start px-4">
            <div class="flex max-w-[min(48rem,100%)] items-start gap-2 rounded-2xl border border-gray-700 bg-gray-800/80 px-3 py-2 text-xs leading-relaxed text-gray-400">
              <span class="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400 animate-ping"></span>
              <div class="min-w-0 flex-1">
                <div class="flex items-start gap-2">
                  <span class="min-w-0 whitespace-normal break-words text-gray-300">
                    {activeStatus()}
                  </span>
                  <div class="ml-auto shrink-0 flex items-center gap-1.5 text-gray-500">
                    <Show when={perfMetrics() && perfMetrics()!.current_tps > 0}>
                      <span
                        class="font-mono text-[10px]"
                        classList={{
                          "text-amber-400 font-semibold": perfMetrics()!.is_cpu_bound,
                          "text-gray-400": !perfMetrics()!.is_cpu_bound,
                        }}
                        title="Instantaneous token generation speed"
                      >
                        {perfMetrics()!.current_tps.toFixed(1)} t/s
                      </span>
                      <span class="text-gray-600">•</span>
                    </Show>
                    <span class="tabular-nums">
                      {formatElapsed(statusElapsedMs())}
                    </span>
                  </div>
                </div>
              </div>
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
        <div class="flex items-center justify-between max-w-5xl mx-auto mb-2 px-1">
          <button
            type="button"
            onClick={toggleAutoApply}
            class={`flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium rounded-full border transition-all ${
              autoApply()
                ? "bg-emerald-950/50 border-emerald-700/60 text-emerald-400 hover:bg-emerald-900/60"
                : "bg-gray-900 border-gray-700 text-gray-400 hover:bg-gray-800 hover:text-gray-300"
            }`}
            title={
              autoApply()
                ? "Direct Edits: approved patches are automatically written to disk"
                : "Review Mode: patches require manual approval before writing to disk"
            }
          >
            <span>{autoApply() ? "mode: direct edits" : "mode: review edits"}</span>
          </button>
        </div>

        <form onSubmit={handleSubmit} class="relative max-w-5xl mx-auto">
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
        <div class="flex items-center justify-between max-w-3xl mx-auto mt-2 text-xs text-gray-600 px-1">
          <p>Shift + Enter for newline.</p>
          <Show when={props.indexingStatus === "indexing"}>
            <span class="flex items-center gap-1.5 text-amber-400/90 text-[11px] animate-pulse">
              <span class="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
              Indexing workspace in background...
            </span>
          </Show>
        </div>
      </div>
    </div>
  );
}
