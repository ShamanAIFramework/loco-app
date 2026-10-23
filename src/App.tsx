import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { createSignal, onMount, Show, For } from "solid-js";
import AiChat from "./AiChat";
import GitPanel from "./GitPanel";
import NotebookPanel from "./NotebookPanel";
import { NotebookEntryView } from "./types/notebook";

import "./App.css";

interface AiConnectionDetails {
  state: string;
  port: number;
  api_key: string;
}

export interface SessionDetails {
  id: string;
  title: string | null;
  workspace_path: string | null;
  model_name: string | null;
  created_at?: string;
  updated_at?: string;
}

function formatTimestamp(timestamp?: string): string {
  if (!timestamp) return "";
  try {
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return timestamp;
    return date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return timestamp;
  }
}

export default function App() {
  const [appState, setAppState] = createSignal("booting");
  const [, setConnDetails] =
    createSignal<AiConnectionDetails | null>(null);
  const [activeSession, setActiveSession] = createSignal<SessionDetails | null>(
    null,
  );
  const [conversations, setConversations] = createSignal<SessionDetails[]>([]);
  const [indexingStatus, setIndexingStatus] = createSignal<"idle" | "indexing" | "ready">("idle");
  const [isCreatingSession, setIsCreatingSession] = createSignal(false);
  const [sessionError, setSessionError] = createSignal<string | null>(null);
  const [notebookEntries, setNotebookEntries] = createSignal<NotebookEntryView[]>([]);
  const [leftPanelTab, setLeftPanelTab] = createSignal<"notebook" | "chats" | "workspace">("notebook");

  // If null or undefined, the backend defaults to the active repository working directory
  const workspacePath: string | null = null;

  const handleNotebookUpdate = (entries: NotebookEntryView[]) => {
    setNotebookEntries(entries);
    if (entries.length > 0) {
      setLeftPanelTab("notebook");
    }
  };

  const fetchConversations = async (): Promise<SessionDetails[]> => {
    try {
      const list = await invoke<SessionDetails[]>("list_conversations");
      setConversations(list);
      return list;
    } catch (e) {
      console.error("Failed to list conversations:", e);
      return [];
    }
  };

  const createNewSession = async () => {
    if (isCreatingSession()) return;

    setIsCreatingSession(true);
    setSessionError(null);
    try {
      const session = await invoke<SessionDetails>("create_session", {
        workspacePath,
      });
      setActiveSession(session);
      setNotebookEntries([]);
      await fetchConversations();
    } catch (e) {
      console.error("Failed to create session:", e);
      setSessionError(String(e));
    } finally {
      setIsCreatingSession(false);
    }
  };

  const switchSession = (session: SessionDetails) => {
    setActiveSession(session);
    setNotebookEntries([]);
  };

  const triggerReindex = async () => {
    setIndexingStatus("indexing");
    try {
      await invoke("trigger_index_workspace", {
        workspacePath: activeSession()?.workspace_path || workspacePath,
      });
    } catch (e) {
      console.error("Failed to trigger re-index:", e);
      setIndexingStatus("idle");
    }
  };

  const initializeSession = async () => {
    if (activeSession() || isCreatingSession()) return;

    setIsCreatingSession(true);
    setSessionError(null);
    try {
      const existing = await fetchConversations();
      if (existing && existing.length > 0) {
        // Select most recent conversation
        setActiveSession(existing[0]);
        // Trigger background indexing for this workspace without blocking
        invoke("trigger_index_workspace", {
          workspacePath: existing[0].workspace_path || workspacePath,
        }).catch((err) => console.error("Initial background indexing error:", err));
      } else {
        // First-time setup: create initial conversation
        const session = await invoke<SessionDetails>("create_session", {
          workspacePath,
        });
        setActiveSession(session);
        await fetchConversations();
      }
    } catch (e) {
      console.error("Failed to initialize session:", e);
      setSessionError(String(e));
    } finally {
      setIsCreatingSession(false);
    }
  };

  const fetchConnectionDetails = async () => {
    const details = await invoke<AiConnectionDetails | null>(
      "get_ai_connection",
    );
    if (details) {
      setConnDetails(details);
      setAppState(details.state);

      if (details.state === "ready") {
        await initializeSession();
      }
    }
  };

  onMount(async () => {
    await listen<string>("ai-state", async (event) => {
      const newState = event.payload;
      setAppState(newState);
      if (newState === "ready") {
        await fetchConnectionDetails();
      }
    });

    await listen<string>("workspace-index-status", (event) => {
      setIndexingStatus(event.payload as "indexing" | "ready");
      if (event.payload === "ready") {
        fetchConversations();
      }
    });

    await fetchConnectionDetails();
  });

  return (
    <div class="flex h-screen w-full bg-gray-950 text-white overflow-hidden">
      {/* Left Panel: Workspace & Session Details */}
      <div class="w-[22vw] min-w-[240px] max-w-[340px] bg-gray-900 border-r border-gray-800 flex flex-col h-full overflow-hidden">
        {/* Header with New Session button */}
        <div class="p-3.5 border-b border-gray-800 flex items-center justify-between shrink-0">
          <div>
            <h1 class="font-bold text-gray-200 tracking-wide text-sm flex items-center gap-1.5">
              LOCO
              <Show when={indexingStatus() === "indexing"}>
                <span class="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" title="Workspace indexing in background"></span>
              </Show>
            </h1>
            <p class="text-[10px] text-gray-500">Local-First AI Agent</p>
          </div>
          <button
            onClick={createNewSession}
            disabled={isCreatingSession()}
            class="flex items-center gap-1 px-2.5 py-1 text-xs font-medium bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-md transition-colors cursor-pointer shadow-sm"
            title="Start a new conversation"
          >
            <span>+</span>
            <span>New Chat</span>
          </button>
        </div>

        {/* Modular Panel Slot Navigation */}
        <div class="flex items-center border-b border-gray-800 bg-gray-950/40 px-2 pt-1 shrink-0">
          <button
            onClick={() => setLeftPanelTab("notebook")}
            class={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
              leftPanelTab() === "notebook"
                ? "border-blue-500 text-white bg-gray-800/40"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            <span>📓 Notebook</span>
            <Show when={notebookEntries().length > 0}>
              <span class="px-1.5 py-0.2 rounded-full bg-blue-900/60 text-blue-300 font-mono text-[10px]">
                {notebookEntries().length}
              </span>
            </Show>
          </button>
          <button
            onClick={() => setLeftPanelTab("chats")}
            class={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
              leftPanelTab() === "chats"
                ? "border-blue-500 text-white bg-gray-800/40"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            <span>💬 Chats</span>
            <Show when={conversations().length > 0}>
              <span class="px-1.5 py-0.2 rounded-full bg-gray-800 text-gray-400 font-mono text-[10px]">
                {conversations().length}
              </span>
            </Show>
          </button>
          <button
            onClick={() => setLeftPanelTab("workspace")}
            class={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
              leftPanelTab() === "workspace"
                ? "border-blue-500 text-white bg-gray-800/40"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            <span>⚙️ Workspace</span>
            <Show when={indexingStatus() === "indexing"}>
              <span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span>
            </Show>
          </button>
        </div>

        {/* Panel Content Slot */}
        <div class="flex-1 min-h-0 overflow-hidden flex flex-col p-2">
          {/* Tab 1: Notebook */}
          <Show when={leftPanelTab() === "notebook"}>
            <NotebookPanel entries={notebookEntries()} />
          </Show>

          {/* Tab 2: Chats History */}
          <Show when={leftPanelTab() === "chats"}>
            <div class="flex-1 flex flex-col overflow-y-auto space-y-1.5 pr-1">
              <Show
                when={conversations().length > 0}
                fallback={
                  <div class="p-4 text-center text-xs text-gray-500">
                    No conversation history found. Click "+ New Chat" to begin.
                  </div>
                }
              >
                <For each={conversations()}>
                  {(conv) => {
                    const isSelected = () => activeSession()?.id === conv.id;
                    return (
                      <button
                        onClick={() => switchSession(conv)}
                        class={`w-full text-left p-2.5 rounded-lg border text-xs transition-colors cursor-pointer flex flex-col gap-1 ${
                          isSelected()
                            ? "bg-blue-950/40 border-blue-600/60 text-white"
                            : "bg-gray-800/30 border-gray-800 hover:bg-gray-800/70 text-gray-300"
                        }`}
                      >
                        <div class="flex items-center justify-between w-full">
                          <span class="font-medium truncate max-w-[170px]" title={conv.title || "Untitled Chat"}>
                            {conv.title || "Untitled Chat"}
                          </span>
                          <Show when={isSelected()}>
                            <span class="text-[9px] px-1.5 py-0.5 rounded bg-blue-600 text-white font-mono">Active</span>
                          </Show>
                        </div>
                        <div class="flex items-center justify-between text-[10px] text-gray-500">
                          <span class="truncate max-w-[120px]">
                            {conv.workspace_path ? conv.workspace_path.split("/").pop() : "Workspace"}
                          </span>
                          <span>{formatTimestamp(conv.created_at)}</span>
                        </div>
                      </button>
                    );
                  }}
                </For>
              </Show>
            </div>
          </Show>

          {/* Tab 3: Workspace & Engine Settings */}
          <Show when={leftPanelTab() === "workspace"}>
            <div class="flex-1 p-3 space-y-5 overflow-y-auto">
              <Show
                when={activeSession()}
                fallback={
                  <div class="animate-pulse flex flex-col gap-2">
                    <div class="h-4 bg-gray-800 rounded w-3/4"></div>
                    <div class="h-3 bg-gray-800 rounded w-1/2"></div>
                  </div>
                }
              >
                <div>
                  <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                    Active Workspace
                  </h2>
                  <div class="bg-gray-800/50 p-2.5 rounded-lg border border-gray-700/50">
                    <p
                      class="text-xs text-gray-300 font-mono whitespace-normal break-all leading-relaxed"
                      title={activeSession()?.workspace_path || "None"}
                    >
                      {activeSession()?.workspace_path || "No workspace"}
                    </p>
                  </div>
                </div>

                <div>
                  <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                    Codebase Index
                  </h2>
                  <div class="bg-gray-800/50 p-2.5 rounded-lg border border-gray-700/50 flex flex-col gap-2.5">
                    <div class="flex items-center justify-between">
                      <span class="text-xs text-gray-400">Status</span>
                      <Show when={indexingStatus() === "indexing"}>
                        <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-amber-950/60 text-amber-300 border border-amber-800/50 animate-pulse">
                          <span class="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                          Indexing...
                        </span>
                      </Show>
                      <Show when={indexingStatus() === "ready"}>
                        <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-emerald-950/60 text-emerald-300 border border-emerald-800/50">
                          <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                          Indexed & Watching
                        </span>
                      </Show>
                      <Show when={indexingStatus() === "idle"}>
                        <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-gray-800 text-gray-400">
                          Ready
                        </span>
                      </Show>
                    </div>

                    <button
                      onClick={triggerReindex}
                      disabled={indexingStatus() === "indexing"}
                      class="w-full mt-1 py-1.5 px-3 rounded text-xs font-medium bg-gray-700 hover:bg-gray-600 disabled:opacity-50 text-gray-200 transition-colors cursor-pointer border border-gray-600/50 flex items-center justify-center gap-1.5"
                    >
                      <span>🔄</span>
                      <span>{indexingStatus() === "indexing" ? "Indexing Codebase..." : "Re-index Workspace"}</span>
                    </button>
                  </div>
                </div>

                <div>
                  <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                    Engine & Model
                  </h2>
                  <div class="bg-gray-800/50 p-2.5 rounded-lg border border-gray-700/50 space-y-2">
                    <div class="flex items-center justify-between text-xs text-gray-300">
                      <span class="text-gray-400">Engine State</span>
                      <span class="capitalize text-emerald-400 font-medium">{appState()}</span>
                    </div>
                    <div class="flex items-center justify-between text-xs text-gray-300">
                      <span class="text-gray-400">Model</span>
                      <span class="font-mono text-[11px]">{activeSession()?.model_name || "Qwen2.5-Coder"}</span>
                    </div>
                  </div>
                </div>
              </Show>
            </div>
          </Show>
        </div>
      </div>

      {/* Center Panel: Chat Interface */}
      <div class="w-[58vw] flex-1 flex flex-col min-w-0">
        <Show when={appState() !== "ready"}>
          <div class="flex flex-col items-center justify-center h-full">
            <h1 class="text-xl font-medium animate-pulse text-gray-400">
              Initializing Engine...
            </h1>
            <Show when={appState() === "failed"}>
              <p class="text-red-400 mt-2">Failed to start AI engine.</p>
            </Show>
          </div>
        </Show>

        <Show when={appState() === "ready" && activeSession() !== null}>
          <AiChat
            session={activeSession()!}
            onNotebookUpdate={handleNotebookUpdate}
          />
        </Show>

        <Show when={appState() === "ready" && activeSession() === null}>
          <div class="flex flex-col items-center justify-center h-full px-8 text-center">
            <h1 class="text-xl font-medium text-gray-300">
              {isCreatingSession()
                ? "Preparing workspace..."
                : "Workspace session unavailable"}
            </h1>
            <p class="text-sm text-gray-500 mt-2 max-w-xl">
              {isCreatingSession()
                ? "Indexing the local project and starting file watching."
                : sessionError() || "The engine is ready, but no workspace session is active."}
            </p>
            <Show when={!isCreatingSession()}>
              <button
                class="mt-5 px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-200 hover:bg-gray-700"
                onClick={initializeSession}
              >
                Retry session
              </button>
            </Show>
          </div>
        </Show>
      </div>

      {/* Right Panel: Git File State */}
      <Show when={activeSession()?.workspace_path}>
        {(path) => <GitPanel workspacePath={path()} />}
      </Show>
    </div>
  );
}
