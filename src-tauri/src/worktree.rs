// src-tauri/src/worktree.rs
//! Manages isolated Git worktrees for safe, sandboxed code editing and validation.

use anyhow::{anyhow, Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default directory for Loco worktrees: `~/.loco/worktrees`
pub fn default_worktrees_dir() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".loco").join("worktrees")
    } else {
        std::env::temp_dir().join("loco-worktrees")
    }
}

/// Resolves the target worktree path for a given session.
pub fn resolve_worktree_path(session_id: &str, base_dir: Option<&Path>) -> PathBuf {
    let base = match base_dir {
        Some(dir) => dir.to_path_buf(),
        None => default_worktrees_dir(),
    };
    base.join(session_id)
}

/// Ensures an isolated Git worktree exists for the given session.
///
/// If the worktree does not exist:
/// 1. Runs `git worktree add --detach <worktree_path> HEAD` against `project_root`.
/// 2. If `project_root/node_modules` exists, symlinks it into `<worktree_path>/node_modules`.
///
/// Returns the canonical path to the worktree directory.
pub fn ensure_worktree(
    project_root: &Path,
    session_id: &str,
    base_dir: Option<&Path>,
) -> Result<PathBuf> {
    let worktree_path = resolve_worktree_path(session_id, base_dir);

    // If worktree directory already exists and contains a .git file/dir, verify and return
    if worktree_path.exists() {
        let git_marker = worktree_path.join(".git");
        if git_marker.exists() {
            return Ok(worktree_path);
        }
    }

    if let Some(parent) = worktree_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create worktree parent directory: {:?}", parent))?;
    }

    // Run `git worktree add --detach <path> HEAD`
    let output = Command::new("git")
        .current_dir(project_root)
        .args([
            "worktree",
            "add",
            "--detach",
            worktree_path.to_str().ok_or_else(|| anyhow!("Invalid worktree path"))?,
            "HEAD",
        ])
        .output()
        .context("Failed to execute `git worktree add`")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("Failed to create git worktree: {}", stderr.trim()));
    }

    // Link dependency caches (e.g. node_modules) if present in the project root
    link_dependency_caches(project_root, &worktree_path);

    Ok(worktree_path)
}

/// Symlinks dependencies such as `node_modules` into the worktree so tests run with warm caches.
fn link_dependency_caches(project_root: &Path, worktree_path: &Path) {
    let source_nm = project_root.join("node_modules");
    let target_nm = worktree_path.join("node_modules");

    if source_nm.is_dir() && !target_nm.exists() {
        #[cfg(unix)]
        {
            let _ = std::os::unix::fs::symlink(&source_nm, &target_nm);
        }
        #[cfg(windows)]
        {
            let _ = std::os::windows::fs::symlink_dir(&source_nm, &target_nm);
        }
    }
}

/// Computes the unified diff of all changes in the worktree against HEAD and applies them
/// to the primary project root.
///
/// Returns the unified diff string that was applied.
pub fn apply_worktree_changes(project_root: &Path, worktree_path: &Path) -> Result<String> {
    // 1. Generate unified diff from worktree
    let diff_output = Command::new("git")
        .current_dir(worktree_path)
        .args(["diff", "HEAD"])
        .output()
        .context("Failed to compute diff in worktree")?;

    if !diff_output.status.success() {
        let stderr = String::from_utf8_lossy(&diff_output.stderr);
        return Err(anyhow!("Failed to read git diff from worktree: {}", stderr.trim()));
    }

    let diff_str = String::from_utf8_lossy(&diff_output.stdout).to_string();
    if diff_str.trim().is_empty() {
        return Ok(String::new());
    }

    // 2. Apply diff to primary project_root
    let mut apply_child = Command::new("git")
        .current_dir(project_root)
        .args(["apply", "--whitespace=nowarn"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("Failed to spawn `git apply` in project root")?;

    if let Some(mut stdin) = apply_child.stdin.take() {
        stdin
            .write_all(diff_str.as_bytes())
            .context("Failed to write diff to `git apply` stdin")?;
    }

    let apply_output = apply_child
        .wait_with_output()
        .context("Failed to wait for `git apply`")?;

    if !apply_output.status.success() {
        let stderr = String::from_utf8_lossy(&apply_output.stderr);
        return Err(anyhow!("Failed to apply worktree diff to project root: {}", stderr.trim()));
    }

    Ok(diff_str)
}

/// Prunes an isolated Git worktree for the given session.
pub fn prune_worktree(
    project_root: &Path,
    session_id: &str,
    base_dir: Option<&Path>,
) -> Result<()> {
    let worktree_path = resolve_worktree_path(session_id, base_dir);

    if worktree_path.exists() {
        // Attempt clean git removal
        let _ = Command::new("git")
            .current_dir(project_root)
            .args([
                "worktree",
                "remove",
                "--force",
                worktree_path.to_str().unwrap_or_default(),
            ])
            .output();

        // If directory remains (e.g. untracked symlinks), force remove
        if worktree_path.exists() {
            let _ = std::fs::remove_dir_all(&worktree_path);
        }
    }

    // Run worktree prune to clear git metadata
    let _ = Command::new("git")
        .current_dir(project_root)
        .args(["worktree", "prune"])
        .output();

    Ok(())
}

/// Cleans up any orphaned worktree directories that do not belong to active sessions.
pub fn cleanup_orphaned_worktrees(
    project_root: &Path,
    active_session_ids: &[String],
    base_dir: Option<&Path>,
) -> Result<usize> {
    let base = match base_dir {
        Some(dir) => dir.to_path_buf(),
        None => default_worktrees_dir(),
    };

    if !base.exists() {
        return Ok(0);
    }

    let mut pruned_count = 0;
    if let Ok(entries) = std::fs::read_dir(&base) {
        for entry in entries.flatten() {
            if let Ok(file_type) = entry.file_type() {
                if file_type.is_dir() {
                    let folder_name = entry.file_name().to_string_lossy().to_string();
                    if !active_session_ids.iter().any(|id| id == &folder_name) {
                        let _ = prune_worktree(project_root, &folder_name, Some(&base));
                        pruned_count += 1;
                    }
                }
            }
        }
    }

    Ok(pruned_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_temp_git_repo() -> (PathBuf, PathBuf) {
        let id = uuid::Uuid::new_v4().to_string();
        let repo_dir = std::env::temp_dir().join(format!("loco-test-repo-{}", id));
        let worktrees_dir = std::env::temp_dir().join(format!("loco-test-worktrees-{}", id));

        std::fs::create_dir_all(&repo_dir).unwrap();
        std::fs::create_dir_all(&worktrees_dir).unwrap();

        // Run git init
        let run_git = |args: &[&str]| {
            let output = Command::new("git")
                .current_dir(&repo_dir)
                .args(args)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "git {:?} failed: {}",
                args,
                String::from_utf8_lossy(&output.stderr)
            );
        };

        run_git(&["init"]);
        run_git(&["config", "user.name", "Loco Test"]);
        run_git(&["config", "user.email", "test@loco.dev"]);

        // Create an initial commit
        let src_dir = repo_dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(src_dir.join("main.rs"), "fn main() {\n    println!(\"original\");\n}\n").unwrap();

        run_git(&["add", "."]);
        run_git(&["commit", "-m", "Initial commit"]);

        (repo_dir, worktrees_dir)
    }

    #[test]
    fn test_worktree_lifecycle_and_patch_application() {
        let (repo_dir, worktrees_dir) = init_temp_git_repo();
        let session_id = "test-session-1";

        // 1. Ensure worktree
        let worktree_path = ensure_worktree(&repo_dir, session_id, Some(&worktrees_dir)).unwrap();
        assert!(worktree_path.exists());
        assert!(worktree_path.join("src").join("main.rs").exists());

        // 2. Modify file in worktree
        let worktree_main = worktree_path.join("src").join("main.rs");
        std::fs::write(&worktree_main, "fn main() {\n    println!(\"patched in worktree\");\n}\n").unwrap();

        // 3. Apply changes back to repo
        let diff = apply_worktree_changes(&repo_dir, &worktree_path).unwrap();
        assert!(diff.contains("-    println!(\"original\");"));
        assert!(diff.contains("+    println!(\"patched in worktree\");"));

        // Check that repo main.rs now has the patched content
        let repo_main_content = std::fs::read_to_string(repo_dir.join("src").join("main.rs")).unwrap();
        assert_eq!(repo_main_content, "fn main() {\n    println!(\"patched in worktree\");\n}\n");

        // 4. Prune worktree
        prune_worktree(&repo_dir, session_id, Some(&worktrees_dir)).unwrap();
        assert!(!worktree_path.exists());

        // Cleanup temp directories
        let _ = std::fs::remove_dir_all(&repo_dir);
        let _ = std::fs::remove_dir_all(&worktrees_dir);
    }

    #[test]
    fn test_cleanup_orphaned_worktrees() {
        let (repo_dir, worktrees_dir) = init_temp_git_repo();

        // Create two worktrees
        let _wt1 = ensure_worktree(&repo_dir, "session-active", Some(&worktrees_dir)).unwrap();
        let wt2 = ensure_worktree(&repo_dir, "session-abandoned", Some(&worktrees_dir)).unwrap();

        assert!(wt2.exists());

        // Run cleanup with only "session-active" declared active
        let active_sessions = vec!["session-active".to_string()];
        let pruned = cleanup_orphaned_worktrees(&repo_dir, &active_sessions, Some(&worktrees_dir)).unwrap();

        assert_eq!(pruned, 1);
        assert!(!wt2.exists());

        // Cleanup temp directories
        let _ = prune_worktree(&repo_dir, "session-active", Some(&worktrees_dir));
        let _ = std::fs::remove_dir_all(&repo_dir);
        let _ = std::fs::remove_dir_all(&worktrees_dir);
    }
}
