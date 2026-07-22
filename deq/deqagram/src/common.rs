use std::fmt;

use pest::RuleType;
use pest::iterators::Pairs;

#[must_use]
pub fn decode_tag<R: RuleType>(pairs: Pairs<R>) -> String {
    let mut result = String::new();
    for pair in pairs {
        match pair.as_str() {
            "\\C" => result.push(']'),
            "\\B" => result.push('\\'),
            "\\r" => result.push('\r'),
            "\\n" => result.push('\n'),
            s => result.push_str(s),
        }
    }
    result
}

#[must_use]
pub fn encode_tag(tag: Option<&str>) -> String {
    let Some(s) = tag else { return String::new() };
    let mut result = String::from("[");
    for ch in s.chars() {
        match ch {
            ']' => result.push_str("\\C"),
            '\\' => result.push_str("\\B"),
            '\r' => result.push_str("\\r"),
            '\n' => result.push_str("\\n"),
            _ => result.push(ch),
        }
    }
    result.push(']');
    result
}

/// Decodes the escape sequences of a `.deq` string literal's inner text (the
/// content between the surrounding quotes).
///
/// Recognizes `\n`, `\r`, `\t`, `\\`, and `\"`; an unrecognized escape `\x`
/// decodes to the bare character `x` (the backslash is dropped). This mirrors
/// the format's own string decoding and is the inverse of [`encode_string`] for
/// the recognized sequences.
#[must_use]
pub fn decode_string(raw: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            result.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => result.push('\n'),
            Some('r') => result.push('\r'),
            Some('t') => result.push('\t'),
            Some('"') => result.push('"'),
            // A recognized `\\`, an unknown escape `\x` (drop the backslash,
            // keep `x`), or a trailing backslash at end of input.
            Some('\\') | None => result.push('\\'),
            Some(other) => result.push(other),
        }
    }
    result
}

/// Encodes a string as the inner text of a `.deq` string literal, escaping the
/// characters [`decode_string`] recognizes so the value round-trips.
#[must_use]
pub fn encode_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            _ => result.push(ch),
        }
    }
    result
}

/// Writes `items` to `f`, separated by `sep`, using each item's `Display`.
///
/// # Errors
///
/// Returns an error if writing to the formatter fails.
pub fn write_separated<T: fmt::Display>(f: &mut fmt::Formatter<'_>, items: &[T], sep: &str) -> fmt::Result {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            f.write_str(sep)?;
        }
        write!(f, "{item}")?;
    }
    Ok(())
}
