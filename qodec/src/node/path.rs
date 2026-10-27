pub(crate) use super::model::ReferenceSegment as Segment;
use super::model::{MAX_SELECTED_POSITIONS, ReferenceParseError};
use std::fmt;

/// Failure parsing or following a model path.
#[derive(Debug, Clone, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum PathError {
    /// The path does not follow the model-path grammar.
    #[display("invalid model path: {_0:?}")]
    Syntax(#[error(not(source))] String),
    /// A field, key, or index does not exist at this position.
    #[display("model path does not exist: {_0:?}")]
    Missing(#[error(not(source))] String),
}

fn parse_selector(token: &str, atom: &str) -> Result<Segment, ReferenceParseError> {
    let bad_index = |part: &str| ReferenceParseError::BadIndex {
        atom: atom.to_owned(),
        index_token: part.to_owned(),
    };
    if token.contains(',') && !token.contains(':') {
        return token
            .split(',')
            .map(|part| {
                let part = part.trim();
                part.parse::<usize>().map_err(|_| bad_index(part))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Segment::Union);
    }
    if token.contains(':') {
        let parts: Vec<_> = token.split(':').map(str::trim).collect();
        let (first, limit, stride) = match parts.as_slice() {
            [first, limit] => (*first, *limit, "1"),
            [first, limit, stride] => (*first, *limit, *stride),
            _ => return Err(bad_index(token)),
        };
        let first = first.parse::<usize>().map_err(|_| bad_index(first))?;
        let limit = limit.parse::<usize>().map_err(|_| bad_index(limit))?;
        let stride_value = stride.parse::<usize>().map_err(|_| bad_index(stride))?;
        if stride_value == 0 {
            return Err(bad_index(stride));
        }
        if first >= limit {
            return Err(ReferenceParseError::EmptySelection(atom.to_owned()));
        }
        let selected = (limit - first).div_ceil(stride_value);
        if selected > MAX_SELECTED_POSITIONS {
            return Err(ReferenceParseError::SelectionTooLarge {
                atom: atom.to_owned(),
                selected,
            });
        }
        return Ok(Segment::Slice {
            start: first,
            stop: limit,
            step: stride_value,
        });
    }
    token.parse().map(Segment::Index).map_err(|_| bad_index(token))
}

pub(crate) fn indices(segment: &Segment) -> impl Iterator<Item = usize> + '_ {
    let (index, union, range, step): (Option<usize>, &[usize], _, _) = match segment {
        Segment::Index(index) => (Some(*index), &[], 0..0, 1),
        Segment::Union(indices) => (None, indices.as_slice(), 0..0, 1),
        Segment::Slice { start, stop, step } => (None, &[], *start..*stop, *step),
        _ => (None, &[], 0..0, 1),
    };
    index
        .into_iter()
        .chain(union.iter().copied())
        .chain(range.step_by(step))
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ModelPath(pub(crate) Vec<Segment>);

impl ModelPath {
    pub(crate) fn parse_reference(text: &str) -> Result<Self, ReferenceParseError> {
        let invalid = || ReferenceParseError::Unrecognized(text.to_owned());
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
                    let end = bracket.find(']').ok_or_else(invalid)?;
                    let selector = parse_selector(&bracket[..end], text)?;
                    rest = &bracket[end + 1..];
                    segments.push(selector);
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

    pub(crate) fn child(&self, segment: Segment) -> Self {
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
                Segment::Union(indices) => {
                    formatter.write_str("[")?;
                    for (position, index) in indices.iter().enumerate() {
                        if position > 0 {
                            formatter.write_str(",")?;
                        }
                        write!(formatter, "{index}")?;
                    }
                    formatter.write_str("]")?;
                }
                Segment::Slice { start, stop, step } => {
                    write!(formatter, "[{start}:{stop}")?;
                    if *step != 1 {
                        write!(formatter, ":{step}")?;
                    }
                    formatter.write_str("]")?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ModelPath;

    #[test]
    fn reference_segments_preserve_structural_distinctions() {
        use super::super::model::Reference;
        use super::Segment;
        let reference = Reference::parse("layers[00].gadgets[\"name\"][1:5:2][3,1,3]").unwrap();
        assert_eq!(
            reference.segments(),
            &[
                Segment::Field("layers".into()),
                Segment::Index(0),
                Segment::Field("gadgets".into()),
                Segment::Key("name".into()),
                Segment::Slice {
                    start: 1,
                    stop: 5,
                    step: 2
                },
                Segment::Union(vec![3, 1, 3]),
            ]
        );
        assert_ne!(Segment::Field("name".into()), Segment::Key("name".into()));
        assert_eq!(reference.path(), "layers[00].gadgets[\"name\"][1:5:2][3,1,3]");
    }

    #[test]
    fn canonical_paths_round_trip() {
        for path in [
            "",
            "layers[0]",
            "layers[0:2]",
            "layers[2,0,2]",
            r#"layers[0].gadgets["measure_xx"].readouts[0].equation[1]"#,
            r#"metadata["a.b[2]\"\\"]["\u03bb"]"#,
        ] {
            let parsed = ModelPath::parse_reference(path).unwrap();
            assert_eq!(ModelPath::parse_reference(&parsed.to_string()).unwrap(), parsed);
        }
        assert_eq!(
            ModelPath::parse_reference("layers[0002]").unwrap().to_string(),
            "layers[2]"
        );
    }

    #[test]
    fn invalid_paths_are_rejected() {
        for path in [
            ".layers",
            "layers.",
            "layers..name",
            "layers[-1]",
            "layers[0:0]",
            "layers[0:2:0]",
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
            assert!(ModelPath::parse_reference(path).is_err(), "{path}");
        }
    }
}
