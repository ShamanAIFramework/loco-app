import { createSignal, createEffect, For, Show, onCleanup } from "solid-js";
import { fetch } from "@tauri-apps/plugin-http";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

interface AiConnectionDetails {
  state: string;
  port: number;
  api_key: string;
}

const MAX_HISTORY_MESSAGES = 12;
const STREAM_FLUSH_MS = 50;

export default function AiChat(props: { connDetails: AiConnectionDetails }) {
  const [messages, setMessages] = createSignal<Message[]>([
    {
      role: "system",
      content: "You are a helpful coding assistant.",
    },
  ]);

  const [input, setInput] = createSignal("");
  const [isGenerating, setIsGenerating] = createSignal(false);

  const [showDebug, setShowDebug] = createSignal(false);
  const [debugPaused, setDebugPaused] = createSignal(false);

  let chatContainerRef: HTMLDivElement | undefined;

  let scrollPending = false;
  let stopRequested = false;

  const addDebug = (value: unknown) => {
    if (debugPaused()) return;

    console.log("REQUEST", JSON.stringify(value, null, 2));
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
    scheduleScroll();
  });

  onCleanup(() => {
    scrollPending = false;
    stopRequested = true;
  });

  function getConversationWindow(history: Message[]): Message[] {
    return history.slice(-MAX_HISTORY_MESSAGES);
  }

  async function handleSubmit(e?: Event) {
    e?.preventDefault();

    const currentInput = input().trim();

    if (!currentInput || isGenerating()) {
      return;
    }

    stopRequested = false;

    const userMessage: Message = {
      role: "user",
      content: currentInput,
    };

    const updatedMessages = [...messages(), userMessage];

    setMessages(updatedMessages);
    setInput("");
    setIsGenerating(true);

    addDebug("===== NEW REQUEST =====");

    let isContinuing = false;
    let continueCount = 0;
    const MAX_CONTINUES = 5; // Safety ceiling to prevent infinite loops

    try {
      while (!stopRequested) {
        let finishReason: string | null = null;

        // On continuation, pass the current history + a silent continuation instruction
        const payloadMessages = !isContinuing
          ? getConversationWindow(updatedMessages)
          : getConversationWindow([
              ...messages(),
              {
                role: "user",
                content:
                  "Continue generating exactly where you left off. Do not repeat any code or text, do not add conversational commentary, and do not repeat markdown code fences if already open. Continue directly with the next characters.",
              },
            ]);

        const payload = {
          model: "default",
          messages: payloadMessages,
          temperature: 0.1,
          max_tokens: -1,
          stream: true,
        };

        addDebug({
          request: payload,
          isContinuing,
          continueCount,
        });

        const response = await fetch(
          `http://127.0.0.1:${props.connDetails.port}/v1/chat/completions`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${props.connDetails.api_key}`,
            },
            body: JSON.stringify(payload),
          },
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        if (!response.body) {
          throw new Error("No response body returned.");
        }

        // Only create a new assistant message bubble on the FIRST chunk
        if (!isContinuing) {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: "",
            },
          ]);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let assistantBuffer = "";
        let partialData = "";

        const flushBuffer = () => {
          if (!assistantBuffer) return;

          const chunk = assistantBuffer;
          assistantBuffer = "";

          // Appends directly to the latest assistant message (works for continuations too)
          setMessages((prev) => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;

            if (lastIndex >= 0 && updated[lastIndex].role === "assistant") {
              updated[lastIndex] = {
                ...updated[lastIndex],
                content: updated[lastIndex].content + chunk,
              };
            }

            return updated;
          });
        };

        const flushTimer = setInterval(flushBuffer, STREAM_FLUSH_MS);

        try {
          while (true) {
            if (stopRequested) {
              addDebug("STREAM STOPPED BY USER");
              break;
            }

            const { done, value } = await reader.read();

            if (done) {
              addDebug("STREAM COMPLETE");
              break;
            }

            const chunk = decoder.decode(value, { stream: true });
            partialData += chunk;

            const events = partialData.split("\n\n");
            partialData = events.pop() ?? "";

            for (const event of events) {
              if (stopRequested) break;

              const lines = event.split("\n");

              for (const line of lines) {
                const trimmed = line.trim();

                if (!trimmed.startsWith("data:")) continue;

                const dataString = trimmed.slice(5).trim();

                if (!dataString || dataString === "[DONE]") continue;

                try {
                  const data = JSON.parse(dataString);
                  const choice = data?.choices?.[0];

                  // 1. CAPTURE FINISH REASON
                  if (choice?.finish_reason) {
                    finishReason = choice.finish_reason;
                  }

                  if (choice?.delta?.tool_calls) {
                    addDebug({ tool_calls: choice.delta.tool_calls });
                  }

                  if (choice?.delta?.function_call) {
                    addDebug({ function_call: choice.delta.function_call });
                  }

                  const text = choice?.delta?.content ?? choice?.text ?? "";

                  if (typeof text === "string" && text.length > 0) {
                    assistantBuffer += text;
                  }
                } catch (err) {
                  addDebug({
                    parse_error: String(err),
                    raw: dataString,
                  });
                }
              }
            }
          }
        } finally {
          clearInterval(flushTimer);
          flushBuffer();

          try {
            reader.releaseLock();
          } catch {
            // ignore
          }
        }

        // 2. CHECK IF WE SHOULD LOOP OR STOP
        if (
          finishReason === "length" &&
          !stopRequested &&
          continueCount < MAX_CONTINUES
        ) {
          continueCount++;
          isContinuing = true;
          addDebug(
            `Hit length limit. Triggering auto-continue (${continueCount}/${MAX_CONTINUES})...`,
          );
        } else {
          // Finished naturally with "stop" or stopped by user
          break;
        }
      }
    } catch (error) {
      console.error("Inference Error:", error);

      addDebug({
        inference_error: String(error),
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "**Error:** Failed to reach the local engine.",
        },
      ]);
    } finally {
      setIsGenerating(false);
      stopRequested = false;
    }
  }

  const stopGeneration = () => {
    stopRequested = true;
    setIsGenerating(false);
  };

  return (
    <div class="flex flex-col h-full w-full max-w-4xl mx-auto bg-gray-900 border-x border-gray-800">
      <div class="flex items-center justify-between p-4 border-b border-gray-800 bg-gray-900/95 sticky top-0">
        <h2 class="text-sm font-semibold text-gray-200">Local AI Session</h2>

        <div class="flex items-center gap-2">
          <button
            onClick={() => setShowDebug(!showDebug())}
            class="text-xs px-2 py-1 rounded bg-gray-800 border border-gray-700 text-gray-300"
          >
            Debug
          </button>

          <span class="text-xs text-gray-500 font-mono">
            Port: {props.connDetails.port}
          </span>
        </div>
      </div>

      <div
        ref={chatContainerRef}
        class="flex-1 overflow-y-auto p-4 space-y-6 scroll-smooth"
      >
        <For each={messages()}>
          {(msg) => (
            <div
              class={`flex w-full ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                class={`max-w-[80%] rounded-2xl px-5 py-3 ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white rounded-br-sm"
                    : "bg-gray-800 text-gray-200 border border-gray-700 rounded-bl-sm"
                }`}
              >
                <p class="whitespace-pre-wrap text-sm leading-relaxed">
                  {msg.content}
                </p>
              </div>
            </div>
          )}
        </For>

        <Show when={isGenerating()}>
          <div class="flex justify-start">
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

      <div class="p-4 bg-gray-900 border-t border-gray-800">
        <form onSubmit={handleSubmit} class="relative">
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
            class="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded-xl px-4 py-3 pr-14 focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none"
            rows="1"
            disabled={isGenerating()}
          />

          <Show
            when={!isGenerating()}
            fallback={
              <button
                type="button"
                onClick={stopGeneration}
                class="absolute right-2 bottom-2 px-3 py-1 rounded bg-red-600 text-white text-xs"
              >
                Stop
              </button>
            }
          >
            <button
              type="submit"
              disabled={!input().trim()}
              class="absolute right-2 bottom-2 p-2 text-gray-400 hover:text-blue-400 transition-colors"
            >
              Send
            </button>
          </Show>
        </form>

        <p class="text-center text-xs text-gray-500 mt-2">
          Shift + Enter for newline. Debug panel captures raw SSE events.
        </p>
      </div>
    </div>
  );
}
