# Test basic pixpack functionality

# Test string roundtrip
test_string <- "Hello, PixPack! This is a test string with special characters: åäöñç"
temp_png <- tempfile(fileext = ".png")

# Test StringConversion encode/decode
tryCatch(
    {
        # Encode string to PNG
        result_png <- StringConversion(test_string, temp_png)
        expect_true(file.exists(result_png), info = "PNG file should be created")
        expect_equal(tools::file_ext(result_png), "png", info = "Should create PNG file")

        # Decode PNG back to string
        decoded_string <- StringConversion("", result_png)
        expect_equal(decoded_string, test_string, info = "Roundtrip should preserve string")

        # Clean up
        if (file.exists(result_png)) unlink(result_png)
    },
    error = function(e) {
        if (file.exists(temp_png)) unlink(temp_png)
        stop("StringConversion test failed: ", e$message)
    }
)

# Test file roundtrip
test_content <- "This is test file content\nWith multiple lines\nAnd special chars: éüß"
temp_txt <- tempfile(fileext = ".txt")
writeLines(test_content, temp_txt)

tryCatch({
    # Encode file to PNG
    result_png <- fileConversion(temp_txt)
    expect_true(file.exists(result_png), info = "PNG should be created from text file")
    expect_equal(tools::file_ext(result_png), "png", info = "Should create PNG file")

    # Decode PNG back to file
    result_decoded <- fileConversion(result_png)
    expect_true(file.exists(result_decoded), info = "Decoded file should exist")

    # Verify content
    decoded_content <- paste(readLines(result_decoded), collapse = "\n")
    original_content <- paste(readLines(temp_txt), collapse = "\n")
    expect_equal(decoded_content, original_content, info = "File content should be preserved")

    # Clean up
    if (file.exists(result_png)) unlink(result_png)
    if (file.exists(result_decoded)) unlink(result_decoded)
}, error = function(e) {
    stop("fileConversion test failed: ", e$message)
}, finally = {
    if (file.exists(temp_txt)) unlink(temp_txt)
})

# Test error conditions
expect_error(fileConversion("nonexistent.txt"), info = "Should error on missing file")
expect_error(StringConversion("test", ""), info = "Should error on empty path")
