import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { createSignal, onMount, Show } from "solid-js";
import AiChat from "./AiChat";

import "./App.css";

interface AiConnectionDetails {
  state: string;
  port: number;
  api_key: string;
}

export default function App() {
  const [appState, setAppState] = createSignal("booting");
  const [connDetails, setConnDetails] =
    createSignal<AiConnectionDetails | null>(null);

  const fetchConnectionDetails = async () => {
    // Calls the Rust #[tauri::command] fn get_ai_connection
    const details = await invoke<AiConnectionDetails | null>(
      "get_ai_connection",
    );
    if (details) {
      setConnDetails(details);
      setAppState(details.state);
    }
  };

  onMount(async () => {
    // 1. Listen for state changes. The payload from Rust is a lowercase string.
    await listen<string>("ai-state", async (event) => {
      const newState = event.payload;
      setAppState(newState);

      if (newState === "ready") {
        await fetchConnectionDetails();
      }
    });

    // 2. Fetch immediately in case we missed the 'ready' event during startup
    await fetchConnectionDetails();
  });

  return (
    <div class="h-screen w-full bg-gray-900 text-white overflow-hidden">
      <Show when={appState() !== "ready"}>
        <div class="flex flex-col items-center justify-center h-full">
          <h1 class="text-xl font-medium animate-pulse">
            Initializing Local AI...
          </h1>
          <Show when={appState() === "failed"}>
            <p class="text-red-400 mt-2">Failed to start AI engine.</p>
          </Show>
        </div>
      </Show>

      <Show when={appState() === "ready" && connDetails() !== null}>
        <AiChat connDetails={connDetails()!} />
      </Show>
    </div>
  );
}
