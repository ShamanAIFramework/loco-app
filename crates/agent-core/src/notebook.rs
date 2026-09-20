use crate::types::Phase;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntryId(pub String);

impl EntryId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LeaseId(pub String);

impl LeaseId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryKind {
    Fact,
    Inference,
    Decision,
    Question,
    Diagnostic,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryStatus {
    Active,
    Stale,
    Superseded,
    Resolved,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Provenance {
    File {
        path: String,
        symbol: Option<String>,
        range: LineRange,
        lease_id: LeaseId,
    },
    CodeGraph {
        query: String,
        result_ref: String,
    },
    Model {
        call_id: String,
    },
    Tool {
        name: String,
        call_id: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceLease {
    pub id: LeaseId,
    pub path: String,
    pub symbol: Option<String>,
    pub range: LineRange,
    pub content_hash: String,
}

impl EvidenceLease {
    pub fn new(path: String, symbol: Option<String>, range: LineRange, content: &str) -> Self {
        Self {
            id: LeaseId::new(),
            path,
            symbol,
            range,
            content_hash: hash_content(content),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NotebookEntry {
    pub id: EntryId,
    pub kind: EntryKind,
    pub statement: String,
    pub provenance: Provenance,
    pub depends_on: Vec<EntryId>,
    pub phase: Phase,
    pub status: EntryStatus,
    pub tags: BTreeSet<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Notebook {
    entries: BTreeMap<EntryId, NotebookEntry>,
    append_order: Vec<EntryId>,
}

impl Notebook {
    pub fn append_tool_fact(
        &mut self,
        statement: impl Into<String>,
        provenance: Provenance,
        phase: Phase,
        tags: impl IntoIterator<Item = String>,
    ) -> EntryId {
        self.append(EntryKind::Fact, statement, provenance, phase, Vec::new(), tags)
    }

    pub fn append_model_inference(
        &mut self,
        statement: impl Into<String>,
        provenance: Provenance,
        phase: Phase,
        depends_on: Vec<EntryId>,
        tags: impl IntoIterator<Item = String>,
    ) -> EntryId {
        self.append(
            EntryKind::Inference,
            statement,
            provenance,
            phase,
            depends_on,
            tags,
        )
    }

    pub fn append_decision(
        &mut self,
        statement: impl Into<String>,
        provenance: Provenance,
        phase: Phase,
        depends_on: Vec<EntryId>,
        tags: impl IntoIterator<Item = String>,
    ) -> EntryId {
        self.append(
            EntryKind::Decision,
            statement,
            provenance,
            phase,
            depends_on,
            tags,
        )
    }

    pub fn active_entries_for_tags(&self, tags: &[String], limit: usize) -> Vec<&NotebookEntry> {
        let wanted: BTreeSet<&str> = tags.iter().map(String::as_str).collect();
        self.append_order
            .iter()
            .filter_map(|id| self.entries.get(id))
            .filter(|entry| entry.status == EntryStatus::Active)
            .filter(|entry| {
                wanted.is_empty() || entry.tags.iter().any(|tag| wanted.contains(tag.as_str()))
            })
            .take(limit)
            .collect()
    }

    pub fn get(&self, id: &EntryId) -> Option<&NotebookEntry> {
        self.entries.get(id)
    }

    pub fn mark_stale_by_lease(&mut self, lease_id: &LeaseId) {
        let mut stale = BTreeSet::new();
        for (id, entry) in &self.entries {
            if provenance_uses_lease(&entry.provenance, lease_id) {
                stale.insert(id.clone());
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for (id, entry) in &self.entries {
                if stale.contains(id) {
                    continue;
                }
                if entry.depends_on.iter().any(|dep| stale.contains(dep)) {
                    stale.insert(id.clone());
                    changed = true;
                }
            }
        }

        for id in stale {
            if let Some(entry) = self.entries.get_mut(&id) {
                entry.status = EntryStatus::Stale;
            }
        }
    }

    pub fn deterministic_checkpoint(&self) -> Self {
        let mut checkpoint = Self::default();
        for id in &self.append_order {
            if let Some(entry) = self.entries.get(id) {
                if entry.status != EntryStatus::Superseded {
                    checkpoint.entries.insert(id.clone(), entry.clone());
                    checkpoint.append_order.push(id.clone());
                }
            }
        }
        checkpoint
    }

    pub fn iter_in_append_order(&self) -> impl Iterator<Item = &NotebookEntry> {
        self.append_order.iter().filter_map(|id| self.entries.get(id))
    }

    fn append(
        &mut self,
        kind: EntryKind,
        statement: impl Into<String>,
        provenance: Provenance,
        phase: Phase,
        depends_on: Vec<EntryId>,
        tags: impl IntoIterator<Item = String>,
    ) -> EntryId {
        let id = EntryId::new();
        let entry = NotebookEntry {
            id: id.clone(),
            kind,
            statement: statement.into(),
            provenance,
            depends_on,
            phase,
            status: EntryStatus::Active,
            tags: tags.into_iter().collect(),
        };
        self.entries.insert(id.clone(), entry);
        self.append_order.push(id.clone());
        id
    }
}

fn provenance_uses_lease(provenance: &Provenance, lease_id: &LeaseId) -> bool {
    matches!(provenance, Provenance::File { lease_id: id, .. } if id == lease_id)
}

pub fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_cascades_without_deleting_history() {
        let lease = LeaseId("lease-1".to_string());
        let mut notebook = Notebook::default();
        let fact = notebook.append_tool_fact(
            "file contains main",
            Provenance::File {
                path: "src/main.rs".to_string(),
                symbol: None,
                range: LineRange { start: 1, end: 4 },
                lease_id: lease.clone(),
            },
            Phase::Extract,
            vec!["src/main.rs".to_string()],
        );
        let inference = notebook.append_model_inference(
            "main starts the app",
            Provenance::Model {
                call_id: "call-1".to_string(),
            },
            Phase::Extract,
            vec![fact],
            vec!["src/main.rs".to_string()],
        );

        notebook.mark_stale_by_lease(&lease);

        assert_eq!(
            notebook.get(&inference).map(|e| &e.status),
            Some(&EntryStatus::Stale)
        );
        assert_eq!(notebook.iter_in_append_order().count(), 2);
    }
}
