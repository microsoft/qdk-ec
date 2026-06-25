use clap::builder::{StringValueParser, TypedValueParser};
use clap::error::{ContextKind, ContextValue, ErrorKind};

#[derive(Clone)]
pub struct SerdeJsonParser;
impl TypedValueParser for SerdeJsonParser {
    type Value = serde_json::Value;
    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let inner = StringValueParser::new();
        let val = inner.parse_ref(cmd, arg, value)?;
        match serde_json::from_str::<serde_json::Value>(&val) {
            Ok(vector) => Ok(vector),
            Err(error) => {
                let mut err = clap::Error::new(ErrorKind::ValueValidation).with_cmd(cmd);
                if let Some(arg) = arg {
                    err.insert(ContextKind::InvalidArg, ContextValue::String(arg.to_string()));
                }
                err.insert(
                    ContextKind::InvalidValue,
                    ContextValue::String(format!("should be like {{\"a\":1}}, parse error: {error}")),
                );
                Err(err)
            }
        }
    }
}

#[derive(Clone)]
pub struct VecUsizeParser;
impl TypedValueParser for VecUsizeParser {
    type Value = Vec<usize>;
    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let inner = StringValueParser::new();
        let val = inner.parse_ref(cmd, arg, value)?;
        match serde_json::from_str::<Vec<usize>>(&val) {
            Ok(vector) => Ok(vector),
            Err(error) => {
                let mut err = clap::Error::new(ErrorKind::ValueValidation).with_cmd(cmd);
                if let Some(arg) = arg {
                    err.insert(ContextKind::InvalidArg, ContextValue::String(arg.to_string()));
                }
                err.insert(
                    ContextKind::InvalidValue,
                    ContextValue::String(format!("should be like [1,2,3], parse error: {}", error)),
                );
                Err(err)
            }
        }
    }
}

#[derive(Clone)]
pub struct VecF64Parser;
impl TypedValueParser for VecF64Parser {
    type Value = Vec<f64>;
    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let inner = StringValueParser::new();
        let val = inner.parse_ref(cmd, arg, value)?;
        match serde_json::from_str::<Vec<f64>>(&val) {
            Ok(vector) => Ok(vector),
            Err(error) => {
                let mut err = clap::Error::new(ErrorKind::ValueValidation).with_cmd(cmd);
                if let Some(arg) = arg {
                    err.insert(ContextKind::InvalidArg, ContextValue::String(arg.to_string()));
                }
                err.insert(
                    ContextKind::InvalidValue,
                    ContextValue::String(format!("should be like [0.1,0.2,0.3], parse error: {error}")),
                );
                Err(err)
            }
        }
    }
}
