//! Report which kernels this CPU can execute, and optionally assert that a
//! named set of them is present.
//!
//! ```text
//! cargo run --example kernels                     # just list them
//! cargo run --example kernels -- avx2-vnni        # list, and fail if absent
//! ```
//!
//! The assertion form exists for CI. A test that pins a kernel the CPU cannot
//! execute passes *vacuously*: dispatch ignores an impossible pin and scores on
//! whatever it would have chosen anyway. Running this first turns that silent
//! no-op into a failure.

use maxsim_lut::{supported_kernels, warm_up};

fn main() {
    let available: Vec<String> = supported_kernels().iter().map(|k| k.to_string()).collect();

    println!("kernels this CPU can execute: {}", available.join(", "));
    println!("calibration chose: {}", warm_up());

    let required: Vec<String> = std::env::args().skip(1).collect();
    let missing: Vec<&String> = required.iter().filter(|r| !available.contains(r)).collect();

    if !missing.is_empty() {
        eprintln!(
            "error: required kernel(s) not available here: {}",
            missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        );
        std::process::exit(1);
    }
}
