use std::fmt;

/// Failure parsing or following a model path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathError {
    /// The path does not follow the model-path grammar.
    Syntax(String),
    /// A field, key, or index does not exist at this position.
    Missing(String),
}

impl fmt::Display for PathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(path) => write!(formatter, "invalid model path: {path:?}"),
            Self::Missing(path) => write!(formatter, "model path does not exist: {path:?}"),
        }
    }
}

impl std::error::Error for PathError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum Segment {
    Field(String),
    Key(String),
    Index(usize),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub(super) struct ModelPath(pub(super) Vec<Segment>);

impl ModelPath {
    pub(super) fn parse(text: &str) -> Result<Self, PathError> {
        let invalid = || PathError::Syntax(text.to_owned());
        let mut rest = text;
        let mut segments = Vec::new();
        while !rest.is_empty() {
            if let Some(bracket) = rest.strip_prefix('[') {
                if bracket.starts_with('"') {
                    let mut stream = serde_json::Deserializer::from_str(bracket).into_iter::<String>();
                    let key = stream.next().ok_or_else(invalid)?.map_err(|_| invalid())?;
                    rest = bracket[stream.byte_offset()..].strip_prefix(']').ok_or_else(invalid)?;
                    segments.push(Segment::Key(key));
                } else {
                    let length = bracket.bytes().take_while(u8::is_ascii_digit).count();
                    let index = bracket[..length].parse().map_err(|_| invalid())?;
                    rest = bracket[length..].strip_prefix(']').ok_or_else(invalid)?;
                    segments.push(Segment::Index(index));
                }
            } else {
                if !segments.is_empty() {
                    rest = rest.strip_prefix('.').ok_or_else(invalid)?;
                }
                if !rest.starts_with(|character: char| character.is_ascii_alphabetic() || character == '_') {
                    return Err(invalid());
                }
                let length = rest
                    .bytes()
                    .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
                    .count();
                segments.push(Segment::Field(rest[..length].to_owned()));
                rest = &rest[length..];
            }
        }
        Ok(Self(segments))
    }

    pub(super) fn child(&self, segment: Segment) -> Self {
        let mut path = self.clone();
        path.0.push(segment);
        path
    }
}

impl fmt::Display for ModelPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, segment) in self.0.iter().enumerate() {
            match segment {
                Segment::Field(field) => {
                    if position > 0 {
                        formatter.write_str(".")?;
                    }
                    formatter.write_str(field)?;
                }
                Segment::Key(key) => write!(formatter, "[{}]", serde_json::Value::String(key.clone()))?,
                Segment::Index(index) => write!(formatter, "[{index}]")?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ModelPath;

    #[test]
    fn canonical_paths_round_trip() {
        for path in [
            "",
            "layers[0]",
            r#"layers[0].gadgets["measure_xx"].readouts[0].equation[1]"#,
            r#"metadata["a.b[2]\"\\"]["\u03bb"]"#,
        ] {
            let parsed = ModelPath::parse(path).unwrap();
            assert_eq!(ModelPath::parse(&parsed.to_string()).unwrap(), parsed);
        }
        assert_eq!(ModelPath::parse("layers[0002]").unwrap().to_string(), "layers[2]");
    }

    #[test]
    fn invalid_paths_are_rejected() {
        for path in [
            ".layers",
            "layers.",
            "layers..name",
            "layers[-1]",
            "layers[0:2]",
            "layers[*]",
            "layers[0]name",
            "layers[0",
            "layers[]",
            "layers[1.0]",
            "layers[true]",
            "layers [0]",
            "layers.[0]",
            r#"metadata["unterminated]"#,
            "layers[999999999999999999999999999]",
        ] {
            assert!(ModelPath::parse(path).is_err(), "{path}");
        }
    }
}
