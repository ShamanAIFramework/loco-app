import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { createSignal, onMount, Show } from "solid-js";
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
  title: string;
  workspace_path: string | null;
  model_name: string | null;
}

export default function App() {
  const [appState, setAppState] = createSignal("booting");
  const [connDetails, setConnDetails] =
    createSignal<AiConnectionDetails | null>(null);
  const [activeSession, setActiveSession] = createSignal<SessionDetails | null>(
    null,
  );
  const [isCreatingSession, setIsCreatingSession] = createSignal(false);
  const [sessionError, setSessionError] = createSignal<string | null>(null);
  const [notebookEntries, setNotebookEntries] = createSignal<NotebookEntryView[]>([]);
  const [leftPanelTab, setLeftPanelTab] = createSignal<"notebook" | "workspace">("notebook");

  const workspacePath = "/Users/ruckus/workspace/loco/loco-app";

  const handleNotebookUpdate = (entries: NotebookEntryView[]) => {
    setNotebookEntries(entries);
    if (entries.length > 0) {
      setLeftPanelTab("notebook");
    }
  };

  const initializeSession = async () => {
    if (activeSession() || isCreatingSession()) return;

    setIsCreatingSession(true);
    setSessionError(null);
    try {
      const session = await invoke<SessionDetails>("create_session", {
        workspacePath,
      });
      setActiveSession(session);
    } catch (e) {
      console.error("Failed to create session:", e);
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

    await fetchConnectionDetails();
  });

  return (
    <div class="flex h-screen w-full bg-gray-950 text-white overflow-hidden">
      {/* Left Panel: Workspace & Session Details */}
      <div class="w-[20vw] bg-gray-900 border-r border-gray-800 flex flex-col h-full overflow-hidden">
        {/* Header */}
        <div class="p-4 border-b border-gray-800 shrink-0">
          <h1 class="font-bold text-gray-200 tracking-wide">LOCO</h1>
          <p class="text-xs text-gray-500 mt-0.5">Local-First AI Agent</p>
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
            onClick={() => setLeftPanelTab("workspace")}
            class={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
              leftPanelTab() === "workspace"
                ? "border-blue-500 text-white bg-gray-800/40"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            <span>⚙️ Workspace</span>
          </button>
        </div>

        {/* Panel Content Slot */}
        <div class="flex-1 min-h-0 overflow-hidden flex flex-col p-2">
          <Show when={leftPanelTab() === "notebook"}>
            <NotebookPanel entries={notebookEntries()} />
          </Show>

          <Show when={leftPanelTab() === "workspace"}>
            <div class="flex-1 p-3 space-y-6 overflow-y-auto">
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
                    Workspace
                  </h2>
                  <div class="bg-gray-800/50 p-2.5 rounded-lg border border-gray-700/50">
                    <p
                      class="text-sm text-gray-300 font-mono whitespace-normal break-all leading-relaxed"
                      title={activeSession()?.workspace_path || "None"}
                    >
                      {activeSession()?.workspace_path || "No workspace"}
                    </p>
                  </div>
                </div>

                <div>
                  <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                    Model
                  </h2>
                  <div class="flex items-center gap-2 text-sm text-gray-300">
                    <span class="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"></span>
                    {activeSession()?.model_name || "Unknown"}
                  </div>
                </div>
              </Show>
            </div>
          </Show>
        </div>
      </div>

      {/* Center Panel: Chat Interface */}
      <div class="w-[60vw] flex-1 flex flex-col min-w-0">
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
