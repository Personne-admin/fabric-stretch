//! `validate <project-dir>`: 0 for OK/warnings, 1 for run errors,
//! 2 for usage errors, an unreadable project or no runs.

use std::process::ExitCode;

use fabric_data::{validate_run, Project, Severity};

fn main() -> ExitCode {
    let Some(root) = std::env::args().nth(1) else {
        eprintln!("usage: validate <project-dir>");
        return ExitCode::from(2);
    };
    if root == "-h" || root == "--help" {
        println!("usage: validate <project-dir>");
        return ExitCode::SUCCESS;
    }

    let project = match Project::discover(&root) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("failed to scan {root}: {err}");
            return ExitCode::from(2);
        }
    };

    if project.runs.is_empty() && project.load_errors.is_empty() {
        eprintln!("no runs found under {root} (no session.json in any subdirectory)");
        return ExitCode::from(2);
    }

    let mut worst = Severity::Ok;
    for run in &project.runs {
        let report = validate_run(run);
        let severity = report.severity();
        worst = worst.max(severity);
        println!("== {}  [{}]", run.key(), severity);
        println!("   dir: {}", run.dir.display());
        for line in report.to_string().lines() {
            println!("   {line}");
        }
        println!();
    }

    for (dir, err) in &project.load_errors {
        worst = Severity::Error;
        println!("== {}  [ERROR]", dir.display());
        println!("   failed to load: {err}");
        println!();
    }

    let groups = project.grouped();
    println!(
        "{} run(s) in {} group(s), {} load error(s); worst severity: {}",
        project.runs.len(),
        groups.len(),
        project.load_errors.len(),
        worst
    );

    match worst {
        Severity::Error => ExitCode::FAILURE,
        _ => ExitCode::SUCCESS,
    }
}
