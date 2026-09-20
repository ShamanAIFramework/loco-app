import { createSignal, createEffect, onMount, For, Show } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import * as Diff2Html from "diff2html";
import "diff2html/bundles/css/diff2html.min.css";
import "highlight.js/styles/github-dark.css";

interface GitFile {
  path: string;
  status: string;
}

export default function GitPanel(props: { workspacePath: string }) {
  const [files, setFiles] = createSignal<GitFile[]>([]);
  const [selectedFile, setSelectedFile] = createSignal<string | null>(null);
  const [diffHtml, setDiffHtml] = createSignal<string>("");
  const [isLoading, setIsLoading] = createSignal(false);

  const fetchStatus = async () => {
    try {
      const result = await invoke<GitFile[]>("get_git_status", {
        workspacePath: props.workspacePath,
      });
      setFiles(result);

      const currentSelected = selectedFile();
      if (currentSelected && !result.some((f) => f.path === currentSelected)) {
        setSelectedFile(null);
        setDiffHtml("");
      }
      // ----------------------
    } catch (e) {
      console.error("Failed to fetch git status:", e);
    }
  };

  const loadDiff = async (path: string) => {
    setSelectedFile(path);
    setIsLoading(true);
    try {
      const diffStr = await invoke<string>("get_git_diff", {
        workspacePath: props.workspacePath,
        filePath: path,
      });

      if (!diffStr) {
        setDiffHtml(
          "<div class='p-4 text-gray-500'>File is newly untracked or has no text diff available.</div>",
        );
      } else {
        // Generate HTML using diff2html
        const html = Diff2Html.html(diffStr, {
          drawFileList: false,
          matching: "lines",
          outputFormat: "line-by-line",
        });
        setDiffHtml(html);
      }
    } catch (e) {
      console.error("Failed to load diff:", e);
      setDiffHtml(`<div class='text-red-500 p-4'>Error: ${e}</div>`);
    } finally {
      setIsLoading(false);
    }
  };

  onMount(() => {
    fetchStatus();
    // Poll for changes every 10 seconds
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  });

  const getStatusColor = (status: string) => {
    if (status.includes("M")) return "text-yellow-400";
    if (status.includes("A") || status.includes("?")) return "text-emerald-400";
    if (status.includes("D")) return "text-red-400";
    return "text-gray-400";
  };

  return (
    <div class="flex flex-col h-full w-80 bg-gray-900 border-l border-gray-800 text-sm overflow-hidden w-[20vw]">
      {/* Header */}
      <div class="p-4 border-b border-gray-800 flex justify-between items-center bg-gray-950/50">
        <h2 class="font-semibold text-gray-200">Source Control</h2>
        <button
          onClick={fetchStatus}
          class="text-gray-400 hover:text-white"
          title="Refresh"
        >
          <svg
            class="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            ></path>
          </svg>
        </button>
      </div>

      {/* Changed Files List (Top Half) */}
      <div class="shrink-0 max-h-60 overflow-y-auto border-b border-gray-800 p-2 space-y-1">
        <Show when={files().length === 0}>
          <div class="p-4 text-gray-500 text-center italic">
            No local changes.
          </div>
        </Show>
        <For each={files()}>
          {(file) => (
            <button
              onClick={() => loadDiff(file.path)}
              // 1. Static base classes go here
              class="w-full text-left px-3 py-1.5 flex items-center justify-between group transition-colors border-l-2"
              // 2. Reactive toggle classes go here in an object!
              classList={{
                "bg-blue-500/10 border-blue-500 text-white":
                  selectedFile() === file.path,
                "border-transparent hover:bg-gray-800/50 text-gray-300":
                  selectedFile() !== file.path,
              }}
            >
              <span class="truncate block" title={file.path}>
                {file.path}
              </span>

              {/* Note: This template string is fine because file.status doesn't change while this button exists */}
              <span
                class={`font-mono text-xs font-bold ${getStatusColor(file.status)}`}
              >
                {file.status}
              </span>
            </button>
          )}
        </For>
      </div>

      {/* Diff Viewer (Bottom Half) */}
      <div class="flex-1 overflow-y-auto bg-[#0d1117] min-h-0 relative">
        <Show when={isLoading()}>
          <div class="p-4 text-gray-500 animate-pulse">Loading diff...</div>
        </Show>
        <Show when={!isLoading() && selectedFile()}>
          <div class="diff-container min-h-full" innerHTML={diffHtml()}></div>
        </Show>
        <Show when={!isLoading() && !selectedFile()}>
          <div class="p-4 text-gray-600 text-center mt-10">
            Select a file to view changes
          </div>
        </Show>
      </div>

      {/* Global override for diff2html dark mode tweaks */}
      <style>{`
        /* Reset wrapper and main table backgrounds */
        .d2h-wrapper, 
        .d2h-file-wrapper, 
        .d2h-files-diff { 
            background: transparent !important; 
            border: none !important; 
            color: #c9d1d9 !important; 
        }

        .d2h-file-wrapper {
            margin-bottom: 0 !important;
        }
        
        /* Hide redundant file header */
        .d2h-file-header { 
            display: none !important; 
        }
        
        /* Code lines font and background */
        .d2h-code-line, 
        .d2h-code-side-line { 
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important; 
            font-size: 12px !important; 
            background-color: transparent !important;
        }

        /* Line numbers column */
        .d2h-code-linenumber, 
        .d2h-code-side-linenumber {
            background-color: #0d1117 !important;
            color: #6e7681 !important;
            border-color: #30363d !important;
        }

        /* Empty placeholder rows */
        .d2h-emptyplaceholder, 
        .d2h-code-side-emptyplaceholder {
            background-color: #010409 !important;
            border-color: #30363d !important;
        }

        /* Insertions (Green) */
        .d2h-ins,
        .d2h-ins .d2h-code-line-ctn,
        .d2h-ins .d2h-code-line-prefix { 
            background-color: rgba(46, 160, 67, 0.15) !important; 
            color: #c9d1d9 !important;
        }

        /* Chunk info headers (@@ -x,y +x,y @@) */
        .d2h-info,
        .d2h-code-line.d2h-info {
            background-color: rgba(56, 139, 253, 0.1) !important; 
            color: #8b949e !important;
            border-color: #30363d !important;
        }
        
        /* Ensure the line number cell for the header is also dark */
        .d2h-code-linenumber.d2h-info {
            background-color: #0d1117 !important;
            color: #8b949e !important;
            border-color: #30363d !important;
        }
        
        /* Highlight specific word changes in insertions */
        .d2h-ins ins {
            background-color: rgba(46, 160, 67, 0.4) !important;
        }

        /* Deletions (Red) */
        .d2h-del,
        .d2h-del .d2h-code-line-ctn,
        .d2h-del .d2h-code-line-prefix { 
            background-color: rgba(248, 81, 73, 0.15) !important; 
            color: #c9d1d9 !important;
        }
        
        /* Highlight specific word changes in deletions */
        .d2h-del del {
            background-color: rgba(248, 81, 73, 0.4) !important;
        }

        /* Hide the +/- symbols for a cleaner look */
        .d2h-code-line-prefix { 
            display: none !important; 
        }
      `}</style>
    </div>
  );
}
