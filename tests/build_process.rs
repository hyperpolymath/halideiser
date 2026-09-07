// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
//! Execute real CMake and a host C fixture to test process orchestration.
//! This does not certify generated Halide algorithms or their schedules.
use std::{fs, process::Command};

fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("halideiser.toml"),
        "[project]\nname = 'probe'\n[[stages]]\nname = 'blur'\noperation = 'blur'\n[target]\narch = 'x86'\n[pipeline]\ninput-format = 'png'\noutput-format = 'png'\n"
    ).unwrap();
    fs::create_dir_all(dir.path().join("generated/halideiser")).unwrap();
    dir
}

fn invoke(dir: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_halideiser"))
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn build_and_run_execute_and_propagate_exit_status() {
    let dir = fixture();
    let source = dir.path().join("generated/halideiser");
    fs::write(source.join("CMakeLists.txt"),
        "cmake_minimum_required(VERSION 3.22)\nproject(probe C)\nadd_executable(probe_runner probe.c)\n").unwrap();
    fs::write(source.join("probe.c"),
        "#include <string.h>\nint main(int argc, char **argv) { return argc == 2 && strcmp(argv[1], \"two words\") == 0 ? 0 : 7; }\n").unwrap();
    let output = invoke(dir.path(), &["build", "--release"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        invoke(dir.path(), &["run", "--", "two words"])
            .status
            .success()
    );
    assert!(!invoke(dir.path(), &["run"]).status.success());
}

#[test]
fn missing_build_inputs_and_missing_runner_fail() {
    let dir = fixture();
    assert!(!invoke(dir.path(), &["build"]).status.success());
    assert!(!invoke(dir.path(), &["run"]).status.success());
}

#[test]
fn compilation_failure_after_successful_configuration_fails() {
    let dir = fixture();
    let source = dir.path().join("generated/halideiser");
    fs::write(source.join("CMakeLists.txt"),
        "cmake_minimum_required(VERSION 3.22)\nproject(probe C)\nadd_executable(probe_runner broken.c)\n").unwrap();
    fs::write(
        source.join("broken.c"),
        "#error planted compilation failure\n",
    )
    .unwrap();
    assert!(!invoke(dir.path(), &["build"]).status.success());
}
