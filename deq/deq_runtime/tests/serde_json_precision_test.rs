#[test]
fn configuration_round_trip_preserves_arbitrary_precision_numbers() {
    let source = r#"{"decoder":{"large_integer":340282366920938463463374607431768211455,"tiny_weight":1e-400},"coordinator":{"buffer_radius":2}}"#;
    let value: serde_json::Value = serde_json::from_str(source).unwrap();
    assert_eq!(
        value["decoder"]["large_integer"].to_string(),
        "340282366920938463463374607431768211455"
    );
    assert_eq!(value["decoder"]["tiny_weight"].to_string(), "1e-400");
    let encoded = serde_json::to_vec(&value).unwrap();
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&encoded).unwrap(), value);
}
