export interface NotebookEntryView {
  id: string;
  kind: "Fact" | "Inference" | "Decision" | "Question" | "Diagnostic" | string;
  statement: string;
  phase: string;
  status: "Active" | "Stale" | "Superseded" | "Resolved" | string;
  tags: string[];
  file_path: string | null;
  line_range: [number, number] | null;
  symbol: string | null;
  lease_id: string | null;
  depends_on: string[];
}

export interface PanelPlugin {
  id: string;
  title: string;
  icon?: string;
  badge?: () => string | number | null;
  render: () => any;
  defaultLocation?: "left" | "right";
}
