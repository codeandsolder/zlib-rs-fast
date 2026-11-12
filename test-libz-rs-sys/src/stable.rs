use zlib_rs::{
    compress_slice, DeflateConfig, Inflate, InflateConfig, InflateError, InflateFlush, ReturnCode,
    Status,
};

use crate::helpers::uncompress_slice_ng;

#[track_caller]
fn assert_inflates(
    expect_header: bool,
    window_bits: u8,
    input: &[u8],
    expected: &[u8],
) -> Result<(), InflateError> {
    let mut d = Inflate::new(expect_header, window_bits)?;
    let mut out = vec![0u8; expected.len()];
    let status = d.decompress(input, &mut out, InflateFlush::Finish)?;
    assert_eq!(status, Status::StreamEnd);
    assert_eq!(out, expected);
    Ok(())
}

fn input_text() -> String {
    // Has some long sequences that repeat.
    let paragraph = concat!(
        "Tweebuffelsmeteenslagmorsdoodgeskietfontein is 'n plaas wat in die provinsie Noordwes ongeveer 200 km wes van ", 
        "Pretoria lê en amptelik by die Landmeter-Generaal geregistreer is.", 
        "\n\n", 
        "Die naam illustreer die saamflansende kenmerk van 'n Afrikaanse samestelling: ", 
        "al die beskrywende aspekte van 'n konsep kan in die algemeen saamgebind word in een lang woord. ", 
        "'n Ander voorbeeld sou wees wildewaatlemoenkonfytkompetisiebeoordelaarshandleiding of wat van een met so 70 letters: ", 
        "maketaanwaatlemoenkonfyttentoonstellingbeoordelaarshandleidingsboekies. ", 
        "Sulke uiterste gebruik is egter nie algemeen nie, en daar is ooreenstemming dat woorde met koppeltekens geskei word", 
        "indien dit té lank begin raak."
    );

    // We want a reasonably sized input so that the window size actually has some effect.
    paragraph.repeat(64)
}

fn compress(window_bits: i32, level: i32) -> Vec<u8> {
    let mut output = vec![0; 1 << 16];

    let config = DeflateConfig {
        level,
        window_bits,

        ..Default::default()
    };
    let (slice, ret) = compress_slice(&mut output, input_text().as_bytes(), config);
    assert_eq!(ret, ReturnCode::Ok);
    let n = slice.len();
    output.truncate(n);

    output
}

fn raw_wbits_9_lvl_9() -> Vec<u8> {
    compress(-9, 9)
}
fn raw_wbits_15_lvl_9() -> Vec<u8> {
    compress(-15, 9)
}
fn zlib_wbits_9_lvl_9() -> Vec<u8> {
    compress(9, 9)
}
fn zlib_wbits_15_lvl_9() -> Vec<u8> {
    compress(15, 9)
}
fn gzip_wbits_9_lvl_9() -> Vec<u8> {
    compress(16 + 9, 9)
}
fn gzip_wbits_15_lvl_9() -> Vec<u8> {
    compress(16 + 15, 9)
}

#[test]
fn inflate_new_valid_inputs() -> Result<(), InflateError> {
    let expected = input_text();

    assert_inflates(false, 9, &raw_wbits_9_lvl_9(), expected.as_bytes())?;
    assert_inflates(false, 15, &raw_wbits_15_lvl_9(), expected.as_bytes())?;

    assert_inflates(true, 9, &zlib_wbits_9_lvl_9(), expected.as_bytes())?;
    assert_inflates(true, 15, &zlib_wbits_15_lvl_9(), expected.as_bytes())?;

    assert_inflates(true, 16 + 9, &gzip_wbits_9_lvl_9(), expected.as_bytes())?;
    assert_inflates(true, 16 + 15, &gzip_wbits_15_lvl_9(), expected.as_bytes())?;

    assert_inflates(true, 32 + 9, &zlib_wbits_9_lvl_9(), expected.as_bytes())?;
    assert_inflates(true, 32 + 15, &zlib_wbits_15_lvl_9(), expected.as_bytes())?;
    assert_inflates(true, 32 + 9, &gzip_wbits_9_lvl_9(), expected.as_bytes())?;
    assert_inflates(true, 47, &zlib_wbits_15_lvl_9(), expected.as_bytes())?;
    assert_inflates(true, 47, &gzip_wbits_9_lvl_9(), expected.as_bytes())?;

    // Using a window bits value of 0 will read the window bits from the zlib header.
    assert_inflates(true, 0, &zlib_wbits_9_lvl_9(), expected.as_bytes())?;
    assert_inflates(true, 0, &zlib_wbits_15_lvl_9(), expected.as_bytes())?;

    // 32 working is a quirk in the logic, and we don't document that this works.
    assert_inflates(true, 32, &gzip_wbits_9_lvl_9(), expected.as_bytes())?;
    assert_inflates(true, 32, &gzip_wbits_15_lvl_9(), expected.as_bytes())?;
    Ok(())
}

#[test]
fn invalid_inflate_window_configurations_return_stream_error() {
    for (expect_header, bits) in [
        (false, 0u8),
        (false, 7),
        (false, 16),
        (false, u8::MAX),
        (true, 7),
        (true, 16),
        (true, 23),
        (true, 39),
        (true, 48),
        (true, 123),
        (true, u8::MAX),
    ] {
        assert!(
            matches!(
                Inflate::new(expect_header, bits),
                Err(InflateError::StreamError)
            ),
            "unexpectedly accepted: header={expect_header}, bits={bits}"
        );
    }
}

#[test]
fn invalid_full_configurations_return_stream_error() {
    for bits in [-16, -1, 7, 16, 23, 39, 48, 64, i32::MIN, i32::MAX] {
        assert!(
            matches!(
                Inflate::new_with_config(InflateConfig { window_bits: bits }),
                Err(InflateError::StreamError)
            ),
            "unexpectedly accepted: {bits}"
        );
    }
}

#[test]
fn raw_stream_is_rejected_in_zlib_autowindow_mode() {
    assert!(assert_inflates(true, 0, &raw_wbits_15_lvl_9(), input_text().as_bytes()).is_err());
}

#[test]
fn gzip_window_bits_0() -> Result<(), InflateError> {
    // This is a quirk of the logic, 32 does work (becomes 0 internally), but 0 itself does not
    // work, due to its "wrap" value being different.
    assert_inflates(true, 32, &gzip_wbits_9_lvl_9(), input_text().as_bytes())?;
    assert_inflates(true, 32, &gzip_wbits_15_lvl_9(), input_text().as_bytes())?;

    let mut output = vec![0u8; 1 << 16];
    let (_, result) = uncompress_slice_ng(
        &mut output,
        &gzip_wbits_9_lvl_9(),
        InflateConfig { window_bits: 0 },
    );
    assert_eq!(result, ReturnCode::DataError);

    assert!(assert_inflates(true, 0, &gzip_wbits_9_lvl_9(), input_text().as_bytes()).is_err());
    Ok(())
}
