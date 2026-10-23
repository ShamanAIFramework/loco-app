import { createSignal, createEffect, onMount, For, Show, createMemo } from "solid-js";
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
  const [rawDiff, setRawDiff] = createSignal<string>("");
  const [branch, setBranch] = createSignal<string>("HEAD");
  const [isLoading, setIsLoading] = createSignal(false);
  const [isRefreshing, setIsRefreshing] = createSignal(false);
  const [copied, setCopied] = createSignal(false);
  const [outputFormat, setOutputFormat] = createSignal<"line-by-line" | "side-by-side">("line-by-line");

  const fetchBranch = async () => {
    try {
      const b = await invoke<string>("get_git_branch", {
        workspacePath: props.workspacePath,
      });
      setBranch(b || "HEAD");
    } catch {
      setBranch("HEAD");
    }
  };

  const fetchStatus = async () => {
    setIsRefreshing(true);
    try {
      await fetchBranch();
      const result = await invoke<GitFile[]>("get_git_status", {
        workspacePath: props.workspacePath,
      });
      setFiles(result);

      const currentSelected = selectedFile();
      if (currentSelected && !result.some((f) => f.path === currentSelected)) {
        setSelectedFile(null);
        setDiffHtml("");
        setRawDiff("");
      } else if (!currentSelected && result.length > 0) {
        // Auto-select first changed file for better UX
        loadDiff(result[0].path);
      }
    } catch (e) {
      console.error("Failed to fetch git status:", e);
    } finally {
      setIsRefreshing(false);
    }
  };

  const renderDiff = (diffStr: string, format: "line-by-line" | "side-by-side") => {
    if (!diffStr || !diffStr.trim()) {
      return "<div class='p-6 text-gray-500 text-center text-xs'>No differences detected.</div>";
    }
    try {
      return Diff2Html.html(diffStr, {
        drawFileList: false,
        matching: "lines",
        outputFormat: format,
      });
    } catch (e) {
      console.error("Failed to parse diff with Diff2Html:", e);
      return `<pre class='p-4 text-xs font-mono text-gray-300 overflow-x-auto'>${diffStr}</pre>`;
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

      setRawDiff(diffStr || "");
      setDiffHtml(renderDiff(diffStr, outputFormat()));
    } catch (e) {
      console.error("Failed to load diff:", e);
      setDiffHtml(`<div class='text-red-400 p-4 text-xs'>Error loading diff: ${e}</div>`);
      setRawDiff("");
    } finally {
      setIsLoading(false);
    }
  };

  const toggleOutputFormat = () => {
    const nextFormat = outputFormat() === "line-by-line" ? "side-by-side" : "line-by-line";
    setOutputFormat(nextFormat);
    if (rawDiff()) {
      setDiffHtml(renderDiff(rawDiff(), nextFormat));
    }
  };

  const handleCopyDiff = async () => {
    if (!rawDiff()) return;
    try {
      await navigator.clipboard.writeText(rawDiff());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy diff:", err);
    }
  };

  onMount(() => {
    fetchStatus();
    // Poll for changes every 10 seconds
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  });

  const getStatusBadge = (status: string) => {
    if (status.includes("M")) {
      return { label: "MOD", color: "bg-yellow-500/15 text-yellow-400 border-yellow-500/30" };
    }
    if (status.includes("A")) {
      return { label: "ADD", color: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30" };
    }
    if (status.includes("?") || status.includes("U")) {
      return { label: "NEW", color: "bg-cyan-500/15 text-cyan-400 border-cyan-500/30" };
    }
    if (status.includes("D")) {
      return { label: "DEL", color: "bg-red-500/15 text-red-400 border-red-500/30" };
    }
    return { label: status, color: "bg-gray-700/30 text-gray-400 border-gray-600/30" };
  };

  // Compute addition and deletion stats for the currently selected file
  const diffStats = createMemo(() => {
    if (!rawDiff()) return { additions: 0, deletions: 0 };
    let additions = 0;
    let deletions = 0;
    for (const line of rawDiff().split("\n")) {
      if (line.startsWith("+") && !line.startsWith("+++")) additions++;
      else if (line.startsWith("-") && !line.startsWith("---")) deletions++;
    }
    return { additions, deletions };
  });

  return (
    <div class="flex flex-col h-full w-80 md:w-96 min-w-[280px] bg-gray-900 border-l border-gray-800 text-sm overflow-hidden select-none">
      {/* Panel Header */}
      <div class="p-3 border-b border-gray-800 flex justify-between items-center bg-gray-950/70">
        <div class="flex items-center gap-2 min-w-0">
          <svg
            class="w-4 h-4 text-purple-400 shrink-0"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <circle cx="18" cy="18" r="3" />
            <circle cx="6" cy="6" r="3" />
            <path d="M18 15V9a9 9 0 0 0-9-9" />
            <path d="M6 9v12" />
          </svg>
          <span class="font-semibold text-gray-200 text-xs">Source Control</span>
          <span class="px-1.5 py-0.2 rounded font-mono text-[11px] font-medium bg-gray-800 text-gray-400 border border-gray-700/60 truncate max-w-[100px]" title={branch()}>
            {branch()}
          </span>
          <Show when={files().length > 0}>
            <span class="px-1.5 py-0.2 rounded-full font-mono text-[10px] font-bold bg-blue-500/20 text-blue-300">
              {files().length}
            </span>
          </Show>
        </div>

        <button
          onClick={fetchStatus}
          disabled={isRefreshing()}
          class="p-1 rounded text-gray-400 hover:text-white hover:bg-gray-800 transition-colors disabled:opacity-50"
          title="Refresh Git status"
        >
          <svg
            class={`w-3.5 h-3.5 ${isRefreshing() ? "animate-spin text-blue-400" : ""}`}
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

      {/* Changed Files List (Top Section) */}
      <div class="shrink-0 max-h-56 overflow-y-auto border-b border-gray-800 p-1.5 space-y-0.5 bg-gray-950/20">
        <Show when={files().length === 0}>
          <div class="py-6 px-4 text-center">
            <svg
              class="w-6 h-6 mx-auto mb-2 text-emerald-400/60"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
            >
              <polyline points="20 6 9 17 4 12" />
            </svg>
            <div class="text-xs font-medium text-gray-400">Working tree clean</div>
            <div class="text-[11px] text-gray-600 mt-0.5">No changes detected</div>
          </div>
        </Show>

        <For each={files()}>
          {(file) => {
            const badge = getStatusBadge(file.status);
            const fileName = () => file.path.split("/").pop() || file.path;
            const dirName = () => {
              const idx = file.path.lastIndexOf("/");
              return idx !== -1 ? file.path.substring(0, idx + 1) : "";
            };

            return (
              <button
                type="button"
                onClick={() => loadDiff(file.path)}
                class="w-full text-left px-2.5 py-1.5 rounded-lg flex items-center justify-between group transition-colors border text-xs"
                classList={{
                  "bg-blue-500/15 border-blue-500/40 text-white font-medium":
                    selectedFile() === file.path,
                  "border-transparent hover:bg-gray-800/60 text-gray-300":
                    selectedFile() !== file.path,
                }}
              >
                <div class="flex items-center gap-1.5 min-w-0 pr-2">
                  <span class="truncate block font-mono text-[11px]" title={file.path}>
                    <span class="text-gray-500">{dirName()}</span>
                    <span class={selectedFile() === file.path ? "text-white font-semibold" : "text-gray-200"}>
                      {fileName()}
                    </span>
                  </span>
                </div>

                <span
                  class={`font-mono text-[10px] font-bold px-1.5 py-0.5 rounded border shrink-0 ${badge.color}`}
                >
                  {badge.label}
                </span>
              </button>
            );
          }}
        </For>
      </div>

      {/* Diff Viewer Area (Bottom Section) */}
      <div class="flex-1 flex flex-col min-h-0 bg-[#0d1117] relative">
        <Show when={selectedFile()}>
          {/* Diff Toolbar */}
          <div class="px-3 py-2 border-b border-gray-800 bg-gray-900/60 flex items-center justify-between shrink-0">
            <div class="flex items-center gap-2 min-w-0">
              <span class="font-mono text-xs text-gray-300 truncate" title={selectedFile()!}>
                {selectedFile()!.split("/").pop()}
              </span>
              <Show when={diffStats().additions > 0 || diffStats().deletions > 0}>
                <div class="flex items-center gap-1 font-mono text-[10px] font-semibold shrink-0">
                  <Show when={diffStats().additions > 0}>
                    <span class="text-emerald-400">+{diffStats().additions}</span>
                  </Show>
                  <Show when={diffStats().deletions > 0}>
                    <span class="text-red-400">-{diffStats().deletions}</span>
                  </Show>
                </div>
              </Show>
            </div>

            <div class="flex items-center gap-1.5 shrink-0">
              {/* Output format toggle: Line-by-line vs Side-by-side */}
              <button
                type="button"
                onClick={toggleOutputFormat}
                title={`Switch to ${outputFormat() === "line-by-line" ? "Side-by-side" : "Unified"} view`}
                class="px-2 py-0.5 text-[10px] rounded border border-gray-700 bg-gray-800/70 hover:bg-gray-700 text-gray-300 transition-colors"
              >
                {outputFormat() === "line-by-line" ? "Split" : "Unified"}
              </button>

              {/* Copy raw diff button */}
              <button
                type="button"
                onClick={handleCopyDiff}
                disabled={!rawDiff()}
                title="Copy raw diff"
                class="px-2 py-0.5 text-[10px] rounded border border-gray-700 bg-gray-800/70 hover:bg-gray-700 text-gray-300 transition-colors disabled:opacity-40"
              >
                {copied() ? "Copied!" : "Copy"}
              </button>
            </div>
          </div>
        </Show>

        {/* Diff Content */}
        <div class="flex-1 overflow-y-auto overflow-x-auto min-h-0 relative">
          <Show when={isLoading()}>
            <div class="p-6 text-gray-500 text-center text-xs animate-pulse">
              Loading diff...
            </div>
          </Show>

          <Show when={!isLoading() && selectedFile()}>
            <div class="diff-container min-h-full" innerHTML={diffHtml()}></div>
          </Show>

          <Show when={!isLoading() && !selectedFile()}>
            <div class="p-8 text-gray-600 text-center mt-6 text-xs">
              <svg
                class="w-8 h-8 mx-auto mb-2 text-gray-700"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
              >
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
                <polyline points="10 9 9 9 8 9" />
              </svg>
              Select a file to inspect diff
            </div>
          </Show>
        </div>
      </div>
    </div>
  );
}
