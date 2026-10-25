import { createSignal, createMemo, Show } from "solid-js";
import * as Diff2Html from "diff2html";
import "diff2html/bundles/css/diff2html.min.css";

interface InlineDiffViewerProps {
  diff: string;
  filePath?: string;
  initialExpanded?: boolean;
  applied?: boolean;
  onApply?: () => void;
}

export default function InlineDiffViewer(props: InlineDiffViewerProps) {
  const [isExpanded, setIsExpanded] = createSignal(
    props.initialExpanded !== undefined ? props.initialExpanded : true,
  );
  const [copied, setCopied] = createSignal(false);

  // Compute addition and deletion line counts from unified diff
  const stats = createMemo(() => {
    let additions = 0;
    let deletions = 0;
    const lines = props.diff.split("\n");
    for (const line of lines) {
      if (line.startsWith("+") && !line.startsWith("+++")) {
        additions++;
      } else if (line.startsWith("-") && !line.startsWith("---")) {
        deletions++;
      }
    }
    return { additions, deletions };
  });

  // Extract display file path if not passed
  const displayPath = createMemo(() => {
    if (props.filePath) return props.filePath;
    const lines = props.diff.split("\n");
    for (const line of lines) {
      if (line.startsWith("diff --git a/")) {
        const parts = line.split(" b/");
        if (parts.length > 1) return parts[1].trim();
      } else if (line.startsWith("+++ b/")) {
        return line.slice(6).trim();
      } else if (line.startsWith("+++ ")) {
        return line.slice(4).trim();
      }
    }
    return "Proposed Changes";
  });

  // Generate syntax-highlighted line-by-line diff HTML
  const diffHtml = createMemo(() => {
    if (!props.diff || !props.diff.trim()) return "";
    try {
      return Diff2Html.html(props.diff, {
        drawFileList: false,
        matching: "lines",
        outputFormat: "line-by-line",
      });
    } catch (e) {
      console.error("Diff2Html inline parsing error:", e);
      return `<pre class="p-3 text-xs font-mono text-gray-300 overflow-x-auto">${props.diff}</pre>`;
    }
  });

  const handleCopy = async (e: MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(props.diff);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy diff:", err);
    }
  };

  const hasChanges = createMemo(() => {
    return stats().additions > 0 || stats().deletions > 0;
  });

  return (
    <Show when={hasChanges()}>
      <div class="w-full my-3 rounded-xl border border-gray-800 bg-[#0d1117] overflow-hidden shadow-sm">
        {/* Diff Header Bar */}
        <div
          onClick={() => setIsExpanded(!isExpanded())}
          class="flex items-center justify-between px-3.5 py-2.5 bg-gray-900/90 border-b border-gray-800/80 cursor-pointer select-none hover:bg-gray-850 transition-colors"
        >
        <div class="flex items-center gap-2 min-w-0">
          {/* File / Git Diff Icon */}
          <svg
            class="w-4 h-4 text-emerald-400 shrink-0"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>

          {/* Path name */}
          <span class="font-mono text-xs font-medium text-gray-200 truncate">
            {displayPath()}
          </span>

          {/* Stat Badges */}
          <div class="flex items-center gap-1.5 ml-1 shrink-0 font-mono text-[11px] font-semibold">
            <Show when={stats().additions > 0}>
              <span class="px-1.5 py-0.2 rounded bg-emerald-500/15 text-emerald-400">
                +{stats().additions}
              </span>
            </Show>
            <Show when={stats().deletions > 0}>
              <span class="px-1.5 py-0.2 rounded bg-red-500/15 text-red-400">
                -{stats().deletions}
              </span>
            </Show>
          </div>
        </div>

        {/* Action Controls */}
        <div class="flex items-center gap-2 shrink-0">
          <Show when={props.applied}>
            <span class="px-2 py-0.5 rounded bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 font-medium text-xs flex items-center gap-1 select-none">
              <svg class="w-3 h-3 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span>Applied to disk</span>
            </span>
          </Show>

          <Show when={!props.applied && props.onApply}>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                props.onApply?.();
              }}
              class="px-2.5 py-1 text-xs font-semibold rounded bg-emerald-600 hover:bg-emerald-500 text-white flex items-center gap-1.5 transition-colors shadow-sm"
              title="Apply this diff to disk"
            >
              <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span>Apply</span>
            </button>
          </Show>

          <button
            type="button"
            title="Copy diff to clipboard"
            onClick={handleCopy}
            class="px-2 py-1 text-xs rounded border border-gray-700 bg-gray-800/60 hover:bg-gray-700 text-gray-300 hover:text-white transition-colors flex items-center gap-1"
          >
            <Show
              when={copied()}
              fallback={
                <>
                  <svg
                    class="w-3.5 h-3.5 text-gray-400"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                  >
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
                  </svg>
                  <span>Copy</span>
                </>
              }
            >
              <svg
                class="w-3.5 h-3.5 text-emerald-400"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span class="text-emerald-400">Copied!</span>
            </Show>
          </button>

          {/* Expand / Collapse Chevron */}
          <svg
            class={`w-4 h-4 text-gray-400 transition-transform duration-200 ${
              isExpanded() ? "rotate-180" : ""
            }`}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      </div>

      {/* Diff Table Body */}
      <Show when={isExpanded()}>
        <div class="overflow-x-auto max-h-[32rem] overflow-y-auto">
          <div class="diff-container min-w-full" innerHTML={diffHtml()}></div>
        </div>
      </Show>
      </div>
    </Show>
  );
}
