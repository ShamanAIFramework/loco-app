import { For, Show, createSignal, createEffect, onCleanup } from "solid-js";
import { marked } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/github-dark.css";
import * as Diff2Html from "diff2html";
import "diff2html/bundles/css/diff2html.min.css";
import InlineDiffViewer from "./InlineDiffViewer";

// Configure marked to support GitHub Flavored Markdown (tables, etc.) and line breaks
marked.setOptions({
  gfm: true,
  breaks: true,
});

export interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  sources?: SourceRef[];
  diff?: string;
  applied?: boolean;
  onApply?: () => void;
}

export interface SourceRef {
  path: string;
  start_line: number;
  end_line: number;
  symbol?: string | null;
  label: string;
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
            fallback={
              <AssistantBubble
                content={msg.content}
                sources={msg.sources}
                diff={msg.diff}
                applied={msg.applied}
                onApply={msg.onApply}
              />
            }
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

function AssistantBubble(props: {
  content: string;
  sources?: SourceRef[];
  diff?: string;
  applied?: boolean;
  onApply?: () => void;
}) {
  let contentRef: HTMLDivElement | undefined;

  // Re-run markdown parsing, highlighting, and button injection when content streams
  createEffect(() => {
    if (!contentRef) return;

    // 1. Parse the markdown string into HTML
    const rawHtml = marked.parse(props.content || "") as string;
    contentRef.innerHTML = rawHtml;

    // 2. Find all <pre> tags
    const preBlocks = contentRef.querySelectorAll("pre");

    preBlocks.forEach((pre) => {
      const codeBlock = pre.querySelector("code");
      let textToCopy = "";

      if (codeBlock) {
        textToCopy = codeBlock.textContent || "";
        const isDiff =
          codeBlock.classList.contains("language-diff") ||
          codeBlock.classList.contains("language-patch");

        if (isDiff && (textToCopy.includes("---") || textToCopy.includes("@@"))) {
          try {
            const renderedHtml = Diff2Html.html(textToCopy, {
              drawFileList: false,
              matching: "lines",
              outputFormat: "line-by-line",
            });
            pre.className =
              "d2h-inline-container rounded-xl border border-gray-800 bg-[#0d1117] overflow-x-auto my-3 !p-0";
            pre.innerHTML = `<div class="diff-container min-w-full text-xs">${renderedHtml}</div>`;
          } catch (e) {
            hljs.highlightElement(codeBlock as HTMLElement);
          }
        } else {
          hljs.highlightElement(codeBlock as HTMLElement);
        }
      }

      // 3. Create a wrapper div to isolate positioning from Tailwind Prose
      const wrapper = document.createElement("div");
      // 'relative' anchors the button. 'my-4' keeps standard vertical spacing.
      wrapper.className = "relative group my-4";
      wrapper.style.position = "relative";

      // Strip vertical margins from the <pre> so it doesn't double-up inside the wrapper
      pre.classList.add("!my-0");

      // 4. Wrap the <pre> tag
      pre.parentNode?.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);

      // 5. Create the copy button
      const btn = document.createElement("button");
      btn.setAttribute("type", "button");
      btn.setAttribute("title", "Copy to clipboard");
      // Uses a slight background opacity (/90) so scrolling code shows slightly underneath it
      btn.style.position = "absolute";
      btn.style.top = "1ch";
      btn.style.right = "1ch";
      btn.style.background = "transparent";
      btn.style.width = "24px";
      btn.style.height = "24px";
      btn.style.border = "none";
      btn.style.borderRadius = "14px";
      btn.style.padding = "0";
      btn.style.margin = "0";
      btn.style.cursor = "pointer";

      const copyIcon = `<svg width="24px" height="24px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M8 4V16C8 17.1046 8.89543 18 10 18L18 18C19.1046 18 20 17.1046 20 16V7.24162C20 6.7034 19.7831 6.18789 19.3982 5.81161L16.0829 2.56999C15.7092 2.2046 15.2074 2 14.6847 2H10C8.89543 2 8 2.89543 8 4Z" stroke="#ccc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M16 18V20C16 21.1046 15.1046 22 14 22H6C4.89543 22 4 21.1046 4 20V9C4 7.89543 4.89543 7 6 7H8" stroke="#ccc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;
      const checkIcon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-4 h-4 text-emerald-400"><polyline points="20 6 9 17 4 12" /></svg>`;

      btn.innerHTML = copyIcon;

      // 6. Attach the copy logic
      btn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(textToCopy);

          btn.innerHTML = checkIcon;
          setTimeout(() => {
            btn.innerHTML = copyIcon;
          }, 2000);
        } catch (err) {
          console.error("Failed to copy code block:", err);
        }
      });

      // 7. Append the button to the WRAPPER, completely bypassing the <pre> tag
      wrapper.appendChild(btn);
    });
  });

  return (
    <div class="flex w-full justify-start px-4">
      <div class="flex flex-col gap-4 w-full rounded-xl p-2 -ml-2">
        <div
          ref={contentRef}
          class="w-full min-w-0 prose prose-invert prose-sm md:prose-base prose-pre:bg-[#0d1117] prose-pre:border prose-pre:border-gray-800 prose-pre:rounded-xl text-gray-300 max-w-none"
        ></div>

        {/* Dedicated Inline Diff Viewer (when diff is attached to message) */}
        <Show when={props.diff && props.diff.trim().length > 0}>
          {(diff) => (
            <InlineDiffViewer
              diff={diff()}
              applied={props.applied}
              onApply={props.onApply}
            />
          )}
        </Show>

        {/* Referenced Sources Stacked Vertically Below, with Chips Flowing Horizontally */}
        <Show when={props.sources && props.sources.length > 0}>
          <div class="w-full pt-3 border-t border-gray-800/80">
            <div class="text-[11px] font-semibold tracking-wider text-gray-400 uppercase mb-2 flex items-center gap-1.5 select-none">
              <svg
                class="w-3.5 h-3.5 text-blue-400 shrink-0"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
                />
              </svg>
              <span>Referenced Sources ({props.sources!.length})</span>
            </div>
            <div class="flex flex-wrap gap-2">
              <For each={props.sources}>
                {(source) => <SourceChip source={source} />}
              </For>
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
}

function SourceChip(props: { source: SourceRef }) {
  const lineLabel = () =>
    props.source.start_line === props.source.end_line
      ? `L${props.source.start_line}`
      : `L${props.source.start_line}-L${props.source.end_line}`;

  const fileName = () =>
    props.source.path.split("/").pop() || props.source.path;

  return (
    <button
      type="button"
      title={`${props.source.path}:${props.source.start_line}-${props.source.end_line}`}
      onClick={() =>
        navigator.clipboard?.writeText(
          `${props.source.path}:${props.source.start_line}-${props.source.end_line}`,
        )
      }
      class="group max-w-full rounded-lg border border-gray-700 bg-gray-900/70 px-3 py-2 text-left text-xs text-gray-300 transition-colors hover:border-gray-600 hover:bg-gray-800/80"
    >
      <div class="flex max-w-full items-center gap-2">
        <span class="shrink-0 rounded bg-blue-500/10 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-blue-300">
          {lineLabel()}
        </span>
        <span class="min-w-0 truncate font-mono text-gray-200">
          {props.source.path}
        </span>
      </div>
      <div class="mt-1 flex items-center gap-2 text-[11px] text-gray-500">
        <span class="truncate">{props.source.label || fileName()}</span>
        <span class="opacity-0 transition-opacity group-hover:opacity-100">
          Click to copy
        </span>
      </div>
    </button>
  );
}
