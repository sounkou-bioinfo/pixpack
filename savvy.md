DESCRIPTION:
Package: savvy
Title: A Simple Wrapper of 'savvy-cli' Command
Version: 0.0.5
Authors@R: 
    c(person(given = "Hiroaki",
             family = "Yutani",
             role = c("aut", "cre"),
             email = "yutani.ini@gmail.com",
             comment = c(ORCID = "0000-0002-3385-7233"))
    )
Description: Generates the C and R wrappers for the Rust-powered R
        package using the savvy framework.
License: MIT + file LICENSE
URL: https://github.com/yutannihilation/savvy
Encoding: UTF-8
Roxygen: list(markdown = TRUE)
RoxygenNote: 7.3.2
Imports: jsonlite, pkgbuild
Suggests: testthat (>= 3.0.0)
Config/testthat/edition: 3
Repository: https://yutannihilation.r-universe.dev
RemoteUrl: https://github.com/yutannihilation/savvy-helper-R-package
RemoteRef: HEAD
RemoteSha: 17f7992f974c446869f89efaf4a0e3f5de4cc3ad
NeedsCompilation: no
Packaged: 2025-08-09 03:30:57 UTC; root
Author: Hiroaki Yutani [aut, cre] (ORCID:
    <https://orcid.org/0000-0002-3385-7233>)
Maintainer: Hiroaki Yutani <yutani.ini@gmail.com>
Depends: R (>= 4.1.0)
Built: R 4.5.1; ; 2025-08-09 16:33:17 UTC; unix

--------------------------------------------------------------------------------
Function: download_savvy_cli()
Update 'savvy-cli'

Description:

     Update 'savvy-cli'

Usage:

     download_savvy_cli()
     

--------------------------------------------------------------------------------
Function: savvy-package()
A Simple Wrapper of 'savvy-cli' Command

Description:

     Generates the C and R wrappers for the Rust-powered R package
     using the savvy framework.

Author(s):

     *Maintainer*: Hiroaki Yutani <mailto:yutani.ini@gmail.com> (ORCID)

See Also:

     Useful links:

        • <https://github.com/yutannihilation/savvy>


--------------------------------------------------------------------------------
Function: savvy_extract_tests()
Execute savvy-cli extract-tests

Description:

     Execute savvy-cli extract-tests

Usage:

     savvy_extract_tests(path = "./src/rust/")
     
Arguments:

    path: Path to the root of a Rust crate.


--------------------------------------------------------------------------------
Function: savvy_init()
Execute 'savvy-cli init"

Description:

     Execute `savvy-cli init``

Usage:

     savvy_init(path = ".", verbose = TRUE)
     
Arguments:

    path: Path to the root of an R package

 verbose: If ‘TRUE’, show all the output from savvy-cli.


--------------------------------------------------------------------------------
Function: savvy_source()
Compile Rust Code And Load

Description:

     Compile Rust Code And Load

Usage:

     savvy_source(
       code,
       use_cache_dir = TRUE,
       env = parent.frame(),
       dependencies = list(),
       clean = NULL
     )
     
Arguments:

    code: Rust code to compile.

use_cache_dir: If ‘TRUE’, reuse and override the cache dir to avoid
          re-compilation. This is an expert-only option.

     env: The R environment where the R wrapping functions should be
          defined.

dependencies: List of dependencies. (e.g. ‘list(once_cell =
          list(version = "1"))’) at the end of the R session.

   clean: If ‘TRUE’, remove the temporary R package used for
          compilation


--------------------------------------------------------------------------------
Function: savvy_update()
Execute 'savvy-cli update"

Description:

     Execute `savvy-cli update``

Usage:

     savvy_update(path = ".", verbose = TRUE)
     
Arguments:

    path: Path to the root of an R package

 verbose: If ‘TRUE’, show all the output from savvy-cli.


--------------------------------------------------------------------------------
Function: savvy_version()
Execute 'savvy-cli -version"

Description:

     Execute `savvy-cli -version``

Usage:

     savvy_version()
     

--------------------------------------------------------------------------------
Function: .onLoad()
function (libname, pkgname) 
{
    dir.create(savvy_cache_dir(), recursive = TRUE, showWarnings = FALSE)
}

--------------------------------------------------------------------------------
Function: %||%()
function (x, y) 
{
    if (is.null(x)) {
        y
    }
    else {
        x
    }
}

--------------------------------------------------------------------------------
Function: check_savvy_cli()
function () 
{
    use_downloaded <- !isTRUE(getOption("savvy.use_installed_cli"))
    if (use_downloaded && !file.exists(savvy_cli_path())) {
        cat("Downloading savvy-cli binary")
        download_savvy_cli()
    }
}

--------------------------------------------------------------------------------
Function: download_savvy_cli()
function () 
{
    download_tmp_dir <- tempfile()
    extract_tmp_dir <- tempfile()
    on.exit(unlink(download_tmp_dir, recursive = TRUE, force = TRUE), 
        add = TRUE)
    on.exit(unlink(extract_tmp_dir, recursive = TRUE, force = TRUE), 
        add = TRUE)
    dir.create(download_tmp_dir)
    download_url <- get_download_url()
    archive_file <- file.path(download_tmp_dir, basename(download_url))
    utils::download.file(download_url, destfile = archive_file, 
        mode = "wb")
    if (Sys.info()[["sysname"]] == "Windows") {
        utils::unzip(archive_file, exdir = extract_tmp_dir)
        file.copy(file.path(extract_tmp_dir, "savvy-cli.exe"), 
            savvy_cli_path(), overwrite = TRUE)
    }
    else {
        utils::untar(archive_file, exdir = extract_tmp_dir, extras = "--strip-components=1")
        file.copy(file.path(extract_tmp_dir, "savvy-cli"), savvy_cli_path(), 
            overwrite = TRUE)
    }
    invisible(NULL)
}

--------------------------------------------------------------------------------
Function: generate_dependencies_toml()
function (dependencies) 
{
    if (!"savvy" %in% names(dependencies)) {
        dependencies$savvy <- list(version = "*")
    }
    crate_names <- names(dependencies)
    x <- vapply(seq_along(dependencies), function(i) {
        dep <- dependencies[[i]]
        name <- crate_names[i]
        keys <- names(dep)
        values <- vapply(dep, function(x) {
            if (length(x) > 1L) {
                sprintf("[%s]", paste(sprintf("\"%s\"", x), collapse = ", "))
            }
            else {
                sprintf("\"%s\"", as.character(x))
            }
        }, character(1L))
        specifications <- paste(keys, "=", values, collapse = "\n")
        sprintf("[dependencies.%s]\n%s\n", name, specifications)
    }, character(1L))
    paste(x, collapse = "\n")
}

--------------------------------------------------------------------------------
Function: generate_pkg_name()
function () 
{
    loaded_dlls <- names(getLoadedDLLs())
    i <- tmp_pkg_count$i
    i <- i + 1L
    new_name <- sprintf("%s%i", SAVVY_PACKAGE_PREFIX, i)
    while (new_name %in% loaded_dlls) {
        new_name <- sprintf("%s%i", SAVVY_PACKAGE_PREFIX, i)
        i <- i + 1
    }
    tmp_pkg_count$i <- i
    new_name
}

--------------------------------------------------------------------------------
Function: get_download_url()
function () 
{
    latest_release <- get_latest_release()
    os <- Sys.info()[["sysname"]]
    arch <- Sys.info()[["machine"]]
    binary <- switch(os, Windows = "savvy-cli-x86_64-pc-windows-msvc.zip", 
        Linux = if (arch == "x86_64") "savvy-cli-x86_64-unknown-linux-gnu.tar.xz" else "savvy-cli-aarch64-unknown-linux-gnu.tar.xz", 
        Darwin = if (arch == "x86_64") "savvy-cli-x86_64-apple-darwin.tar.xz" else "savvy-cli-aarch64-apple-darwin.tar.xz")
    paste(SAVVY_CLI_URL_BASE, latest_release, binary, sep = "/")
}

--------------------------------------------------------------------------------
Function: get_latest_release()
function () 
{
    jsonlite::read_json("https://api.github.com/repos/yutannihilation/savvy/releases/latest")[["tag_name"]]
}

--------------------------------------------------------------------------------
Function: savvy_cache_dir()
function () 
{
    normalizePath(tools::R_user_dir("savvy", "cache"), mustWork = FALSE)
}

--------------------------------------------------------------------------------
Function: savvy_cli_path()
function () 
{
    bin <- if (Sys.info()[["sysname"]] == "Windows") {
        paste0(SAVVY_CLI_NAME, ".exe")
    }
    else {
        SAVVY_CLI_NAME
    }
    if (isTRUE(getOption("savvy.use_installed_cli"))) {
        bin
    }
    else {
        file.path(savvy_cache_dir(), bin)
    }
}

--------------------------------------------------------------------------------
Function: savvy_extract_tests()
function (path = "./src/rust/") 
{
    check_savvy_cli()
    system2(savvy_cli_path(), args = c("extract-tests", path), 
        stdout = TRUE, stderr = FALSE)
}

--------------------------------------------------------------------------------
Function: savvy_init()
function (path = ".", verbose = TRUE) 
{
    check_savvy_cli()
    out <- if (verbose) 
        ""
    else FALSE
    system2(savvy_cli_path(), args = c("init", path), stdout = out, 
        stderr = out)
    cat("\nPlease run `devtools::document()`\n")
}

--------------------------------------------------------------------------------
Function: savvy_source()
function (code, use_cache_dir = TRUE, env = parent.frame(), dependencies = list(), 
    clean = NULL) 
{
    check_savvy_cli()
    pkg_name <- generate_pkg_name()
    if (isTRUE(use_cache_dir)) {
        clean <- clean %||% FALSE
        dir <- file.path(savvy_cache_dir(), "R-package")
        for (pkg_name_prev in names(getLoadedDLLs())) {
            if (startsWith(pkg_name_prev, SAVVY_PACKAGE_PREFIX)) {
                dyn.unload(file.path(dir, "src", sprintf("%s%s", 
                  pkg_name_prev, .Platform$dynlib.ext)))
            }
        }
    }
    else {
        clean <- clean %||% TRUE
        dir <- tempfile()
    }
    dir.create(dir, showWarnings = FALSE, recursive = TRUE)
    if (isTRUE(clean)) {
        on.exit(unlink(dir, recursive = TRUE), add = TRUE)
    }
    dir.create(file.path(dir, "R"), showWarnings = FALSE)
    file.create(file.path(dir, "NAMESPACE"), showWarnings = FALSE)
    writeLines(sprintf(DESCRIPTION, pkg_name), file.path(dir, 
        "DESCRIPTION"))
    if (!dir.exists(file.path(dir, "src"))) {
        savvy_init(dir, verbose = FALSE)
    }
    writeLines(code, file.path(dir, "src", "rust", "src", "lib.rs"))
    tweak_cargo_toml(file.path(dir, "src", "rust", "Cargo.toml"), 
        dependencies)
    savvy_update(dir)
    pkgbuild::compile_dll(dir)
    dll_file <- file.path(dir, "src", sprintf("%s%s", pkg_name, 
        .Platform$dynlib.ext))
    if (isTRUE(clean)) {
        dll_file_orig <- dll_file
        dll_file <- tempfile(fileext = .Platform$dynlib.ext)
        file.rename(dll_file_orig, dll_file)
    }
    dyn.load(dll_file)
    wrapper_file <- file.path(dir, "R", "000-wrappers.R")
    tweak_wrappers(wrapper_file, pkg_name)
    source(wrapper_file, local = env)
}

--------------------------------------------------------------------------------
Function: savvy_update()
function (path = ".", verbose = TRUE) 
{
    check_savvy_cli()
    out <- if (verbose) 
        ""
    else FALSE
    system2(savvy_cli_path(), args = c("update", path), stdout = out, 
        stderr = out)
    cat("\nPlease run `devtools::document()`\n")
}

--------------------------------------------------------------------------------
Function: savvy_version()
function () 
{
    check_savvy_cli()
    system2(savvy_cli_path(), args = c("--version"))
}

--------------------------------------------------------------------------------
Function: tweak_cargo_toml()
function (path, dependencies) 
{
    spec <- readLines(path)
    idx <- which(startsWith(spec, "[dependencies"))[1]
    if (is.na(idx)) {
        stop("No [depndencies] section found in Cargo.toml")
    }
    spec <- spec[1:idx]
    spec <- c(spec, generate_dependencies_toml(dependencies))
    writeLines(spec, path)
}

--------------------------------------------------------------------------------
Function: tweak_wrappers()
function (path, pkg_name) 
{
    r_code <- readLines(path)
    call_wrapper <- sprintf(".Call_%s", pkg_name)
    r_code <- gsub(".Call", call_wrapper, r_code)
    r_code <- c(r_code, "", sprintf("%s <- function(symbol, ...) {", 
        call_wrapper), "  symbol_string <- deparse(substitute(symbol))", 
        sprintf("  .Call(symbol_string, ..., PACKAGE = \"%s\")", 
            pkg_name), "}")
    writeLines(r_code, path)
}
