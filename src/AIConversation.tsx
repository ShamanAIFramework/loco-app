import { For, Show, createSignal, createEffect, onCleanup } from "solid-js";
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
      if (codeBlock) {
        hljs.highlightElement(codeBlock as HTMLElement);
        // codeBlock.style.borderRadius = "14px";
        // codeBlock.style.maxHeight = "33vh";
        // codeBlock.style.overflow = "auto";
        // codeBlock.style.padding = "4ch";
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
        if (!codeBlock) return;
        try {
          const textToCopy = codeBlock.textContent || "";
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
      <div class="flex gap-4 w-full max-w-4xl rounded-xl p-2 -ml-2">
        <div
          ref={contentRef}
          class="flex-1 min-w-0 prose prose-invert prose-sm md:prose-base prose-pre:bg-[#0d1117] prose-pre:border prose-pre:border-gray-800 prose-pre:rounded-xl text-gray-300 max-w-none"
        ></div>
      </div>
    </div>
  );
}
