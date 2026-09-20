import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { createSignal, onMount, Show } from "solid-js";
import AiChat from "./AiChat";
import GitPanel from "./GitPanel";

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

  const fetchConnectionDetails = async () => {
    const details = await invoke<AiConnectionDetails | null>(
      "get_ai_connection",
    );
    if (details) {
      setConnDetails(details);
      setAppState(details.state);

      // If the engine is ready, initialize our first workspace session
      if (details.state === "ready" && !activeSession()) {
        try {
          // Replace this with your actual absolute path (e.g., "C:\\Users\\Matt\\projects\\loco" or "/home/matt/projects/loco")
          const session = await invoke<SessionDetails>("create_session", {
            workspacePath: "/Users/ruckus/workspace/loco/loco-app",
          });
          setActiveSession(session);
        } catch (e) {
          console.error("Failed to create session:", e);
        }
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
      <div class="w-[20vw] bg-gray-900 border-r border-gray-800 flex flex-col">
        <div class="p-4 border-b border-gray-800">
          <h1 class="font-bold text-gray-200 tracking-wide">LOCO</h1>
          <p class="text-xs text-gray-500 mt-1">Local-First AI Agent</p>
        </div>

        <div class="flex-1 p-4 space-y-6 overflow-y-auto">
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
                  class="text-sm text-gray-300 font-mono truncate"
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
          {/* We now pass the session details instead of the raw connection details */}
          <AiChat session={activeSession()!} />
        </Show>
      </div>

      {/* Right Panel: Git File State */}
      <Show when={activeSession()?.workspace_path}>
        {(path) => <GitPanel workspacePath={path()} />}
      </Show>
    </div>
  );
}
