#' PixPack: R bindings for efficient file encoding into PNG images
#'
#' This package provides R bindings for the PixPack utility, which can encode
#' arbitrary files and strings into PNG images and decode them back to their
#' original form.
#'
#' @keywords internal
"_PACKAGE"

#' Encode or decode files with automatic format detection
#'
#' This is a user-friendly wrapper around the core fileConversion function.
#' It automatically detects whether to encode (file -> PNG) or decode (PNG -> file)
#' based on the input file type.
#'
#' @param file_path Path to the input file. If it's a PNG created by PixPack,
#'   it will be decoded. Otherwise, it will be encoded into a PNG.
#' @param verbose Logical; if TRUE, prints status messages
#'
#' @return Path to the output file (either the created PNG or decoded file)
#' @export
#'
#' @examples
#' \dontrun{
#' # Create a test file
#' test_file <- tempfile(fileext = ".txt")
#' writeLines(c("Hello", "World"), test_file)
#'
#' # Encode to PNG
#' png_file <- pixpack_convert(test_file)
#'
#' # Decode back to original
#' decoded_file <- pixpack_convert(png_file)
#'
#' # Clean up
#' unlink(c(png_file, decoded_file, test_file))
#' }
pixpack_convert <- function(file_path, verbose = TRUE) {
    if (!file.exists(file_path)) {
        stop("File does not exist: ", file_path)
    }

    is_png <- tools::file_ext(file_path) == "png"

    if (verbose) {
        if (is_png) {
            message("Decoding PNG to original file...")
        } else {
            message("Encoding file to PNG...")
        }
    }

    result <- fileConversion(file_path)

    if (verbose) {
        message("Output: ", result)
    }

    invisible(result)
}

#' Encode text to PNG or decode PNG to text
#'
#' User-friendly wrapper for encoding strings to PNG images or decoding them back.
#'
#' @param text Character string to encode (ignored if png_path exists and is a PixPack PNG)
#' @param png_path Path where the PNG should be saved or read from
#' @param verbose Logical; if TRUE, prints status messages
#'
#' @return If encoding: path to created PNG. If decoding: the decoded text.
#' @export
#'
#' @examples
#' \dontrun{
#' # Encode text to PNG
#' png_file <- tempfile(fileext = ".png")
#' result <- pixpack_text(text = "Hello, PixPack!", png_path = png_file)
#'
#' # Decode PNG back to text
#' decoded_text <- pixpack_text(png_path = png_file)
#' print(decoded_text) # Should print "Hello, PixPack!"
#'
#' # Clean up
#' unlink(png_file)
#' }
pixpack_text <- function(text = "", png_path, verbose = TRUE) {
    if (missing(png_path) || png_path == "") {
        stop("png_path must be provided")
    }

    if (file.exists(png_path) && tools::file_ext(png_path) == "png") {
        if (verbose) message("Decoding PNG to text...")
        result <- StringConversion("", png_path)
        if (verbose) message("Decoded ", nchar(result), " characters")
        return(result)
    } else {
        if (text == "") {
            stop("Either provide text to encode or an existing PNG file to decode")
        }
        if (verbose) message("Encoding text to PNG...")
        result <- StringConversion(text, png_path)
        if (verbose) message("Created PNG: ", result)
        return(result)
    }
}

#' Visualize PixPack PNG metadata
#'
#' Creates a simple visualization showing the grid structure and metadata
#' of a PixPack PNG file.
#'
#' @param png_path Path to a PixPack PNG file
#' @param show_grid Logical; if TRUE, overlays a grid showing the data cells
#'
#' @return Invisibly returns a list with metadata (if extractable)
#' @export
#'
#' @examples
#' \dontrun{
#' # Create a test PNG
#' test_text <- "Hello, PixPack!"
#' png_file <- tempfile(fileext = ".png")
#' pixpack_text(test_text, png_file)
#'
#' # Visualize it
#' pixpack_plot(png_file)
#'
#' # Clean up
#' unlink(png_file)
#' }
pixpack_plot <- function(png_path, show_grid = TRUE) {
    if (!file.exists(png_path)) {
        stop("PNG file does not exist: ", png_path)
    }

    if (tools::file_ext(png_path) != "png") {
        stop("File must be a PNG: ", png_path)
    }

    # Try to read the PNG
    tryCatch(
        {
            # Use png package if available, otherwise fallback to basic info
            if (requireNamespace("png", quietly = TRUE)) {
                img <- png::readPNG(png_path)

                # Create the plot
                graphics::par(mfrow = c(1, 1), mar = c(4, 4, 4, 2))

                # Plot the image
                graphics::plot(0, 0,
                    type = "n",
                    xlim = c(0, ncol(img)), ylim = c(0, nrow(img)),
                    xlab = "Width (pixels)", ylab = "Height (pixels)",
                    main = paste("PixPack PNG:", basename(png_path))
                )

                # Display the image
                graphics::rasterImage(img, 0, 0, ncol(img), nrow(img))

                # Add grid if requested
                if (show_grid) {
                    # Estimate grid based on typical PixPack patterns
                    # This is approximate since we don't parse the headers here
                    grid_size <- 8 # Common QR-like grid size

                    graphics::abline(
                        v = seq(0, ncol(img), length.out = grid_size + 1),
                        col = "red", lty = 2, lwd = 0.5
                    )
                    graphics::abline(
                        h = seq(0, nrow(img), length.out = grid_size + 1),
                        col = "red", lty = 2, lwd = 0.5
                    )
                }

                # Add basic info
                file_size <- file.info(png_path)$size
                graphics::mtext(paste("File size:", round(file_size / 1024, 1), "KB"),
                    side = 3, line = 0.5, cex = 0.8
                )

                message("PixPack PNG visualized. Dimensions: ", ncol(img), "x", nrow(img))

                return(invisible(list(
                    width = ncol(img),
                    height = nrow(img),
                    channels = if (length(dim(img)) == 3) dim(img)[3] else 1,
                    file_size = file_size
                )))
            } else {
                # Fallback without png package
                file_size <- file.info(png_path)$size

                message("Install the 'png' package for full visualization capabilities.")
                message("PixPack PNG file: ", basename(png_path))
                message("File size: ", round(file_size / 1024, 1), " KB")

                return(invisible(list(file_size = file_size)))
            }
        },
        error = function(e) {
            warning("Could not read PNG file: ", e$message)
            return(invisible(NULL))
        }
    )
}

#' Get information about a PixPack file
#'
#' Extracts basic information about a file or PixPack PNG without decoding.
#'
#' @param file_path Path to the file to inspect
#'
#' @return A list with file information
#' @export
#'
#' @examples
#' \dontrun{
#' # Create test file
#' test_file <- tempfile(fileext = ".txt")
#' writeLines(c("Line 1", "Line 2"), test_file)
#'
#' # Get info
#' info <- pixpack_info(test_file)
#' print(info)
#'
#' # Clean up
#' unlink(test_file)
#' }
pixpack_info <- function(file_path) {
    if (!file.exists(file_path)) {
        stop("File does not exist: ", file_path)
    }

    file_info <- file.info(file_path)
    is_png <- tools::file_ext(file_path) == "png"

    result <- list(
        path = file_path,
        name = basename(file_path),
        size_bytes = file_info$size,
        size_kb = round(file_info$size / 1024, 2),
        modified = file_info$mtime,
        is_png = is_png,
        extension = tools::file_ext(file_path)
    )

    if (is_png) {
        result$type <- "Possibly PixPack PNG (use pixpack_plot for visualization)"
    } else {
        result$type <- "Regular file (can be encoded to PixPack PNG)"
    }

    class(result) <- "pixpack_info"
    return(result)
}

#' @export
print.pixpack_info <- function(x, ...) {
    cat("PixPack File Info:\n")
    cat("  File:", x$name, "\n")
    cat("  Size:", x$size_kb, "KB\n")
    cat("  Type:", x$type, "\n")
    cat("  Modified:", format(x$modified), "\n")
    invisible(x)
}
