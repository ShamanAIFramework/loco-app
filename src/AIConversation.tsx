import { For, Show, createSignal, createEffect } from "solid-js";
import { marked } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/github-dark.css";

// Configure marked to support GitHub Flavored Markdown (tables, etc.) and line breaks
marked.setOptions({
  gfm: true,
  breaks: true,
});

export interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

interface AIConversationProps {
  messages: Message[];
}

export default function AIConversation(props: AIConversationProps) {
  const visibleMessages = () =>
    props.messages.filter((m) => m.role !== "system");

  return (
    <div class="flex flex-col space-y-8 pb-4">
      <For each={visibleMessages()}>
        {(msg) => (
          <Show
            when={msg.role === "user"}
            fallback={<AssistantBubble content={msg.content} />}
          >
            <UserBubble content={msg.content} />
          </Show>
        )}
      </For>
    </div>
  );
}

function UserBubble(props: { content: string }) {
  const [isExpanded, setIsExpanded] = createSignal(false);

  return (
    <div class="flex w-full justify-end px-4">
      <div class="max-w-[75%] group flex flex-col items-end">
        <div
          onClick={() => setIsExpanded(!isExpanded())}
          class="bg-gray-800 border border-gray-700 text-gray-200 rounded-2xl rounded-tr-sm px-5 py-3 cursor-pointer hover:bg-gray-750 transition-colors"
        >
          <p
            class={`whitespace-pre-wrap text-sm leading-relaxed ${
              isExpanded() ? "" : "line-clamp-3"
            }`}
          >
            {props.content}
          </p>
        </div>

        <Show
          when={
            props.content.split("\n").length > 3 || props.content.length > 150
          }
        >
          <span class="text-[10px] text-gray-500 mt-1 mr-1 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer">
            {isExpanded() ? "Click to collapse" : "Click to expand"}
          </span>
        </Show>
      </div>
    </div>
  );
}

function AssistantBubble(props: { content: string }) {
  let contentRef: HTMLDivElement | undefined;

  // Re-run markdown parsing and highlighting every time content streams in
  createEffect(() => {
    if (!contentRef) return;

    // Parse the markdown string into HTML
    const rawHtml = marked.parse(props.content || "") as string;
    contentRef.innerHTML = rawHtml;

    // Manually trigger highlight.js on any newly generated code blocks
    const codeBlocks = contentRef.querySelectorAll("pre code");
    codeBlocks.forEach((block) => {
      hljs.highlightElement(block as HTMLElement);
    });
  });

  return (
    <div class="flex w-full justify-start px-4">
      <div class="flex gap-4 w-full max-w-4xl">
        {/* The ref allows solid to target this div for raw HTML injection */}
        <div
          ref={contentRef}
          class="flex-1 min-w-0 prose prose-invert prose-sm md:prose-base prose-pre:bg-[#0d1117] prose-pre:border prose-pre:border-gray-800 prose-pre:rounded-xl text-gray-300 max-w-none"
        ></div>
      </div>
    </div>
  );
}
