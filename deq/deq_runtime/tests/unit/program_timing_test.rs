use super::*;

fn gadgets() -> Vec<bin::Gadget> {
    [10, 10, 30]
        .into_iter()
        .map(|completion| bin::Gadget {
            gtype: 1,
            metadata: timing_metadata(completion, 0),
            ..Default::default()
        })
        .collect()
}

fn types() -> Vec<bin::GadgetType> {
    vec![bin::GadgetType {
        gtype: 1,
        measurements: vec![bin::gadget_type::Measurement::default()],
        ..Default::default()
    }]
}

#[test]
fn timed_program_retains_equal_and_empty_completions() {
    let mut types = types();
    types.push(bin::GadgetType {
        gtype: 2,
        ..Default::default()
    });
    let mut gadgets = gadgets();
    gadgets[2].gtype = 2;
    let timing = validate(&gadgets, &types).unwrap().unwrap();
    assert_eq!(timing.cumulative_measurements, vec![1, 2, 2]);
    assert_eq!(
        timing.completions,
        vec![Duration::from_nanos(10), Duration::from_nanos(10), Duration::from_nanos(30)]
    );
    for gadget in &mut gadgets {
        gadget.metadata = None;
    }
    assert!(validate(&gadgets, &types).unwrap().is_none());
}

#[test]
fn rejects_corrupt_program_timing_and_ids() {
    let cases: [fn(&mut [bin::Gadget]); 8] = [
        |gadgets| gadgets[1].metadata = None,
        |gadgets| gadgets[1].metadata = timing_metadata(5, 0),
        |gadgets| gadgets[1].metadata = timing_metadata(10, u64::MAX),
        |gadgets| gadgets[1].metadata = timing_metadata(1 << 63, 0),
        |gadgets| gadgets[1].gtype = 42,
        |gadgets| gadgets[1].gid = 1,
        |gadgets| gadgets[1].gid = u64::MAX,
        |gadgets| gadgets[1].connectors.push(bin::gadget::Connector { gid: 1, port: 0 }),
    ];
    for invalidate in cases {
        let mut gadgets = gadgets();
        invalidate(&mut gadgets);
        assert!(validate(&gadgets, types().iter()).is_err());
    }
}

#[test]
fn validates_connector_and_remote_readout_references() {
    let mut types = types();
    types[0].outputs.push(bin::gadget_type::Port {
        ptype: 1,
        ..Default::default()
    });
    types[0].readouts.push(bin::gadget_type::Readout::default());
    types.push(bin::GadgetType {
        gtype: 2,
        inputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        ..Default::default()
    });
    let mut gadgets = gadgets();
    gadgets.truncate(2);
    gadgets[1].gtype = 2;
    gadgets[1].connectors.push(bin::gadget::Connector { gid: 1, port: 0 });
    gadgets[1].modifier = Some(bin::GadgetModifier {
        remote_conditional_correction: Some(bin::RemoteConditionalCorrection {
            remote_readouts: vec![bin::remote_conditional_correction::RemoteReadout {
                gid: 1,
                readout_index: 0,
            }],
            ..Default::default()
        }),
        ..Default::default()
    });
    validate(&gadgets, &types).unwrap();
    for (gid, port) in [(3, 0), (1, 1)] {
        let mut invalid = gadgets.clone();
        invalid[1].connectors[0] = bin::gadget::Connector { gid, port };
        assert!(validate(&invalid, &types).is_err());
    }
    let mut invalid = gadgets.clone();
    invalid[1].metadata = timing_metadata(9, 1);
    assert!(validate(&invalid, &types).is_err());
    let mut invalid = gadgets.clone();
    invalid[1]
        .modifier
        .as_mut()
        .unwrap()
        .remote_conditional_correction
        .as_mut()
        .unwrap()
        .remote_readouts[0]
        .readout_index = 1;
    assert!(validate(&invalid, &types).is_err());
    types[1].inputs[0].ptype = 2;
    assert!(validate(&gadgets, &types).is_err());
}

#[test]
fn metadata_preserves_exact_nanoseconds_and_ignores_other_tools() {
    let mut gadget = bin::Gadget {
        metadata: timing_metadata(9_007_199_254_740_993, 17),
        ..Default::default()
    };
    assert_eq!(gadget_timing(&gadget).unwrap(), Some((9_007_199_254_740_993, 17)));
    let metadata = gadget.metadata.as_mut().unwrap();
    metadata.fields.insert(
        "custom.tool".into(),
        prost_types::Value {
            kind: Some(Kind::BoolValue(true)),
        },
    );
    let Some(Kind::StructValue(timing)) = &mut metadata.fields.get_mut(TIMING_METADATA_KEY).unwrap().kind else {
        unreachable!()
    };
    timing.fields.remove("duration_ns");
    assert_eq!(gadget_timing(&gadget).unwrap(), Some((9_007_199_254_740_993, 0)));
    gadget.metadata.as_mut().unwrap().fields.remove(TIMING_METADATA_KEY);
    assert_eq!(gadget_timing(&gadget).unwrap(), None);
}

#[test]
fn invalid_metadata_values_are_not_silently_ignored() {
    for kind in [
        Kind::NullValue(0),
        Kind::NumberValue(10.0),
        Kind::BoolValue(true),
        Kind::StringValue("-1".into()),
        Kind::StringValue("1.5".into()),
        Kind::StringValue("\u{0661}".into()),
        Kind::StringValue("9223372036854775808".into()),
    ] {
        let mut gadget = bin::Gadget {
            metadata: timing_metadata(0, 0),
            ..Default::default()
        };
        let Some(Kind::StructValue(timing)) = &mut gadget
            .metadata
            .as_mut()
            .unwrap()
            .fields
            .get_mut(TIMING_METADATA_KEY)
            .unwrap()
            .kind
        else {
            unreachable!()
        };
        timing
            .fields
            .insert("start_ns".into(), prost_types::Value { kind: Some(kind) });
        assert!(gadget_timing(&gadget).is_err());
    }
    let mut gadget = bin::Gadget {
        metadata: timing_metadata(0, 0),
        ..Default::default()
    };
    gadget
        .metadata
        .as_mut()
        .unwrap()
        .fields
        .get_mut(TIMING_METADATA_KEY)
        .unwrap()
        .kind = Some(Kind::NullValue(0));
    assert!(gadget_timing(&gadget).is_err());
}
