import { createSignal, createMemo, For, Show } from "solid-js";
import { NotebookEntryView } from "./types/notebook";

export default function NotebookPanel(props: {
  entries: NotebookEntryView[];
  onSelectEntry?: (entry: NotebookEntryView) => void;
}) {
  const [filterKind, setFilterKind] = createSignal<string>("all");
  const [searchQuery, setSearchQuery] = createSignal("");
  const [expandedId, setExpandedId] = createSignal<string | null>(null);

  const factsCount = createMemo(
    () => props.entries.filter((e) => e.kind === "Fact").length,
  );
  const inferencesCount = createMemo(
    () => props.entries.filter((e) => e.kind === "Inference").length,
  );
  const decisionsCount = createMemo(
    () => props.entries.filter((e) => e.kind === "Decision").length,
  );

  const filteredEntries = createMemo(() => {
    const q = searchQuery().trim().toLowerCase();
    const kind = filterKind();

    return props.entries.filter((entry) => {
      if (kind !== "all" && entry.kind.toLowerCase() !== kind.toLowerCase()) {
        return false;
      }

      if (!q) return true;

      const inStatement = entry.statement.toLowerCase().includes(q);
      const inPath = entry.file_path?.toLowerCase().includes(q) ?? false;
      const inSymbol = entry.symbol?.toLowerCase().includes(q) ?? false;
      const inTags = entry.tags.some((t) => t.toLowerCase().includes(q));

      return inStatement || inPath || inSymbol || inTags;
    });
  });

  const getKindBadgeClass = (kind: string) => {
    switch (kind.toLowerCase()) {
      case "fact":
        return "bg-blue-500/15 text-blue-400 border-blue-500/30";
      case "inference":
        return "bg-emerald-500/15 text-emerald-400 border-emerald-500/30";
      case "decision":
        return "bg-purple-500/15 text-purple-400 border-purple-500/30";
      default:
        return "bg-gray-700/40 text-gray-300 border-gray-600/40";
    }
  };

  const getKindIcon = (kind: string) => {
    switch (kind.toLowerCase()) {
      case "fact":
        return "📦";
      case "inference":
        return "💡";
      case "decision":
        return "🎯";
      default:
        return "📋";
    }
  };

  return (
    <div class="flex flex-col h-full w-full bg-gray-900/60 border border-gray-800 rounded-xl overflow-hidden text-xs">
      {/* Panel Title & Total Stats */}
      <div class="p-3 border-b border-gray-800/80 bg-gray-950/60 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="text-sm font-semibold text-gray-200">Notebook</span>
          <span class="px-1.5 py-0.5 rounded-full bg-gray-800 text-gray-400 font-mono text-[10px]">
            {props.entries.length}
          </span>
        </div>
        <Show when={props.entries.length > 0}>
          <div class="flex items-center gap-1.5 text-[10px]">
            <span class="text-blue-400 font-mono">{factsCount()}f</span>
            <span class="text-gray-600">•</span>
            <span class="text-emerald-400 font-mono">{inferencesCount()}i</span>
            <Show when={decisionsCount() > 0}>
              <>
                <span class="text-gray-600">•</span>
                <span class="text-purple-400 font-mono">{decisionsCount()}d</span>
              </>
            </Show>
          </div>
        </Show>
      </div>

      {/* Filter Tabs & Quick Search */}
      <Show when={props.entries.length > 0}>
        <div class="p-2 border-b border-gray-800/60 space-y-2 bg-gray-950/20">
          <div class="flex items-center gap-1 overflow-x-auto pb-0.5">
            <button
              onClick={() => setFilterKind("all")}
              class={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                filterKind() === "all"
                  ? "bg-gray-700 text-white"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              All ({props.entries.length})
            </button>
            <button
              onClick={() => setFilterKind("fact")}
              class={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                filterKind() === "fact"
                  ? "bg-blue-900/50 text-blue-300 border border-blue-700/50"
                  : "text-gray-400 hover:text-blue-400"
              }`}
            >
              Facts ({factsCount()})
            </button>
            <button
              onClick={() => setFilterKind("inference")}
              class={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                filterKind() === "inference"
                  ? "bg-emerald-900/50 text-emerald-300 border border-emerald-700/50"
                  : "text-gray-400 hover:text-emerald-400"
              }`}
            >
              Inferences ({inferencesCount()})
            </button>
            <Show when={decisionsCount() > 0}>
              <button
                onClick={() => setFilterKind("decision")}
                class={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                  filterKind() === "decision"
                    ? "bg-purple-900/50 text-purple-300 border border-purple-700/50"
                    : "text-gray-400 hover:text-purple-400"
                }`}
              >
                Decisions ({decisionsCount()})
              </button>
            </Show>
          </div>

          <div class="relative">
            <input
              type="text"
              placeholder="Filter facts, paths, tags..."
              value={searchQuery()}
              onInput={(e) => setSearchQuery(e.currentTarget.value)}
              class="w-full bg-gray-950/80 border border-gray-800 rounded px-2 py-1 text-[11px] text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500"
            />
            <Show when={searchQuery()}>
              <button
                onClick={() => setSearchQuery("")}
                class="absolute right-2 top-1 text-gray-500 hover:text-gray-300 text-xs"
              >
                ×
              </button>
            </Show>
          </div>
        </div>
      </Show>

      {/* Entry Cards List */}
      <div class="flex-1 overflow-y-auto p-2 space-y-2 min-h-0">
        <Show
          when={props.entries.length > 0}
          fallback={
            <div class="p-6 text-center text-gray-500 space-y-2">
              <span class="text-2xl block">📓</span>
              <p class="font-medium text-gray-400 text-xs">Notebook Ledger Empty</p>
              <p class="text-[11px] text-gray-600 leading-normal">
                Run an agent query in chat to populate active leased facts, code ranges, and verified inferences.
              </p>
            </div>
          }
        >
          <Show
            when={filteredEntries().length > 0}
            fallback={
              <div class="p-4 text-center text-gray-500 italic text-[11px]">
                No entries match filter.
              </div>
            }
          >
            <For each={filteredEntries()}>
              {(entry) => {
                const isExpanded = () => expandedId() === entry.id;
                const toggleExpanded = () =>
                  setExpandedId(isExpanded() ? null : entry.id);

                return (
                  <div
                    class="bg-gray-950/70 border border-gray-800/80 rounded-lg p-2.5 transition-all hover:border-gray-700 space-y-1.5"
                    onClick={() => {
                      props.onSelectEntry?.(entry);
                      toggleExpanded();
                    }}
                  >
                    {/* Header Row: Badge & File Anchor */}
                    <div class="flex items-center justify-between gap-1.5 flex-wrap">
                      <span
                        class={`px-1.5 py-0.5 rounded text-[10px] font-medium border flex items-center gap-1 ${getKindBadgeClass(
                          entry.kind,
                        )}`}
                      >
                        <span>{getKindIcon(entry.kind)}</span>
                        <span>{entry.kind}</span>
                      </span>

                      <Show when={entry.file_path}>
                        <span
                          class="font-mono text-[10px] text-gray-400 bg-gray-900 border border-gray-800 px-1.5 py-0.5 rounded max-w-[140px] truncate"
                          title={`${entry.file_path}:${
                            entry.line_range
                              ? `${entry.line_range[0]}-${entry.line_range[1]}`
                              : ""
                          }`}
                        >
                          {entry.file_path?.split("/").pop()}
                          <Show when={entry.line_range}>
                            <span class="text-gray-500 font-sans">
                              :{entry.line_range![0]}-{entry.line_range![1]}
                            </span>
                          </Show>
                        </span>
                      </Show>
                    </div>

                    {/* Statement Text */}
                    <p class="text-gray-300 text-[11px] leading-relaxed break-words font-sans">
                      {entry.statement}
                    </p>

                    {/* Tags & Metadata */}
                    <Show when={entry.tags.length > 0 || isExpanded()}>
                      <div class="flex items-center gap-1 flex-wrap pt-0.5">
                        <For each={entry.tags}>
                          {(tag) => (
                            <span class="bg-gray-900/90 border border-gray-800 text-gray-500 text-[9px] px-1 py-0.2 rounded font-mono">
                              #{tag}
                            </span>
                          )}
                        </For>
                        <Show when={isExpanded() && entry.depends_on.length > 0}>
                          <span class="text-[9px] text-gray-500 ml-auto font-mono">
                            deps: {entry.depends_on.length}
                          </span>
                        </Show>
                      </div>
                    </Show>
                  </div>
                );
              }}
            </For>
          </Show>
        </Show>
      </div>
    </div>
  );
}
