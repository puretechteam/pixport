//! Builds libjxl, JPEG XL's reference encoder (BSD-3-Clause), as static libraries for jxl.rs.
//! The source is the libjxl tree the jpegxl-src crate ships (or PIXPORT_LIBJXL_SRC); CMake and a
//! C++17 compiler are needed. Only the library is built: no tools, tests, or image-format plugins.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PIXPORT_LIBJXL_SRC");
    let src = libjxl_source();
    let target = env::var("TARGET").unwrap_or_default();

    let mut cfg = cmake::Config::new(&src);
    cfg.profile("Release");
    for (key, value) in [
        ("BUILD_SHARED_LIBS", "OFF"),
        ("BUILD_TESTING", "OFF"),
        ("JPEGXL_ENABLE_TOOLS", "OFF"),
        ("JPEGXL_ENABLE_DEVTOOLS", "OFF"),
        ("JPEGXL_ENABLE_BENCHMARK", "OFF"),
        ("JPEGXL_ENABLE_EXAMPLES", "OFF"),
        ("JPEGXL_ENABLE_FUZZERS", "OFF"),
        ("JPEGXL_ENABLE_DOXYGEN", "OFF"),
        ("JPEGXL_ENABLE_MANPAGES", "OFF"),
        ("JPEGXL_ENABLE_JNI", "OFF"),
        ("JPEGXL_ENABLE_JPEGLI", "OFF"),
        ("JPEGXL_ENABLE_SJPEG", "OFF"),
        ("JPEGXL_ENABLE_OPENEXR", "OFF"),
        ("JPEGXL_ENABLE_PLUGINS", "OFF"),
        ("JPEGXL_ENABLE_TCMALLOC", "OFF"),
        ("JPEGXL_ENABLE_VIEWERS", "OFF"),
        ("JPEGXL_BUNDLE_LIBPNG", "OFF"),
        ("JPEGXL_ENABLE_SKCMS", "ON"),
        ("JPEGXL_ENABLE_TRANSCODE_JPEG", "ON"),
        ("JPEGXL_ENABLE_BOXES", "ON"),
        ("JPEGXL_FORCE_SYSTEM_BROTLI", "OFF"),
        ("JPEGXL_FORCE_SYSTEM_HWY", "OFF"),
        ("JPEGXL_WARNINGS_AS_ERRORS", "OFF"),
        ("HWY_ENABLE_TESTS", "OFF"),
        ("HWY_ENABLE_EXAMPLES", "OFF"),
        ("HWY_ENABLE_CONTRIB", "OFF"),
        ("HWY_ENABLE_INSTALL", "ON"),
    ] {
        cfg.define(key, value);
    }
    if target.contains("msvc") {
        // the same C runtime as the Rust code: static with +crt-static (.cargo/config.toml, so the
        // module needs no msvcp140.dll, which Python doesn't ship), else the DLLs
        let crt_static = env::var("CARGO_CFG_TARGET_FEATURE").is_ok_and(|f| f.split(',').any(|x| x == "crt-static"));
        cfg.define("CMAKE_MSVC_RUNTIME_LIBRARY", if crt_static { "MultiThreaded" } else { "MultiThreadedDLL" });
        cfg.define("CMAKE_POLICY_DEFAULT_CMP0091", "NEW");
        // libjxl 0.12 calls MSVC's _sub_overflow_i32 without including the header that declares it
        cfg.cxxflag("/FIintrin.h");
        // source paths in assertion messages (__FILE__) start at libjxl, not the builder's home
        let trim = format!("/d1trimfile:{}\\", src.display());
        cfg.cflag(&trim).cxxflag(&trim);
    } else {
        // the extension module is a shared library
        cfg.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
        // -ffile-prefix-map is gcc 8+; older cross compilers (manylinux ARM) know only the debug one
        let mut map = format!("-ffile-prefix-map={}=libjxl", src.display());
        if !compiler_accepts(&target, &map) {
            map = format!("-fdebug-prefix-map={}=libjxl", src.display());
        }
        cfg.cflag(&map).cxxflag(&map);
    }
    let dst = cfg.build();

    for dir in ["lib", "lib64"] {
        let d = dst.join(dir);
        if d.is_dir() {
            println!("cargo:rustc-link-search=native={}", d.display());
        }
    }
    for lib in ["jxl", "jxl_cms", "jxl_threads", "hwy", "brotlienc", "brotlidec", "brotlicommon"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

/// Whether the target's C compiler (as the cc crate picks it: CC_<target>, TARGET_CC, CC, or cc)
/// takes `flag`: it preprocesses an empty file with it.
fn compiler_accepts(target: &str, flag: &str) -> bool {
    let cc = [format!("CC_{target}"), format!("CC_{}", target.replace('-', "_")), "TARGET_CC".into(), "CC".into()]
        .iter()
        .find_map(env::var_os)
        .unwrap_or_else(|| "cc".into());
    Command::new(cc)
        .args([flag, "-x", "c", "-E", "-"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// The libjxl source tree: PIXPORT_LIBJXL_SRC, or the one inside the jpegxl-src package.
fn libjxl_source() -> PathBuf {
    if let Some(dir) = env::var_os("PIXPORT_LIBJXL_SRC") {
        return PathBuf::from(dir);
    }
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let manifest = Path::new(&env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let out = Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--offline", "--manifest-path"])
        .arg(&manifest)
        .output()
        .map(|o| (o.stdout, o.stderr))
        .unwrap_or_else(|e| (Vec::new(), e.to_string().into_bytes()));
    let text = String::from_utf8_lossy(&out.0);
    for part in text.split("\"manifest_path\":\"").skip(1) {
        let path = part.split('"').next().unwrap_or("").replace("\\\\", "\\");
        let Some(dir) = Path::new(&path).parent() else { continue };
        let named = dir.file_name().is_some_and(|n| n.to_string_lossy().starts_with("jpegxl-src"));
        if named && dir.join("libjxl").join("CMakeLists.txt").is_file() {
            return dir.join("libjxl");
        }
    }
    // On a fresh machine `cargo metadata --offline` can fail (packages for other platforms aren't
    // downloaded), but cargo has always unpacked the build-dependencies before running this script.
    if let Some(dir) = from_registry() {
        return dir;
    }
    panic!(
        "libjxl's source wasn't found (the jpegxl-src package); set PIXPORT_LIBJXL_SRC to a libjxl checkout. cargo metadata said: {}",
        String::from_utf8_lossy(&out.1).trim()
    );
}

/// libjxl inside the unpacked jpegxl-src package in cargo's registry: CARGO_HOME, or ~/.cargo.
fn from_registry() -> Option<PathBuf> {
    let home = env::var_os("CARGO_HOME").map(PathBuf::from).or_else(|| {
        env::var_os("HOME").or_else(|| env::var_os("USERPROFILE")).map(|h| PathBuf::from(h).join(".cargo"))
    })?;
    let wanted = format!("jpegxl-src-{}", jpegxl_src_version());
    for index in std::fs::read_dir(home.join("registry").join("src")).ok()?.flatten() {
        let dir = index.path().join(&wanted).join("libjxl");
        if dir.join("CMakeLists.txt").is_file() {
            return Some(dir);
        }
    }
    None
}

/// The version Cargo.toml pins jpegxl-src to (`jpegxl-src = "=0.12.0"`), so an older copy in the
/// registry is never picked up.
fn jpegxl_src_version() -> String {
    let manifest = Path::new(&env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let text = std::fs::read_to_string(manifest).expect("can't read Cargo.toml");
    let line = text.lines().find(|l| l.trim_start().starts_with("jpegxl-src")).expect("jpegxl-src in Cargo.toml");
    line.split('"').nth(1).expect("jpegxl-src's version").trim_start_matches('=').to_string()
}
