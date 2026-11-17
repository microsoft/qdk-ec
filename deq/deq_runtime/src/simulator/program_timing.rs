//! Interpret simulator-owned metadata and reconcile it with Stim timing annotations.

use crate::bin;
use crate::simulator::common::DelayBatch;
use prost_types::value::Kind;
use std::collections::HashMap;
use std::time::Duration;

const TIMING_METADATA_KEY: &str = "deq.simulation.timing";

#[derive(Clone, Debug)]
pub(crate) struct ProgramTiming {
    pub completions: Vec<Duration>,
    pub cumulative_measurements: Vec<usize>,
}

fn gadget_timing(gadget: &bin::Gadget) -> Result<Option<(u64, u64)>, String> {
    let Some(value) = gadget
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.fields.get(TIMING_METADATA_KEY))
    else {
        return Ok(None);
    };
    let Some(Kind::StructValue(timing)) = &value.kind else {
        return Err(format!("{TIMING_METADATA_KEY} must be an object"));
    };
    if timing.fields.keys().any(|key| key != "start_ns" && key != "duration_ns") {
        return Err(format!("unknown {TIMING_METADATA_KEY} field"));
    }
    let read = |name: &str| -> Result<u64, String> {
        let Some(value) = timing.fields.get(name) else {
            return if name == "duration_ns" {
                Ok(0)
            } else {
                Err(format!("{TIMING_METADATA_KEY}.{name} must be a decimal string"))
            };
        };
        let Some(Kind::StringValue(text)) = &value.kind else {
            return Err(format!("{TIMING_METADATA_KEY}.{name} must be a decimal string"));
        };
        let value = (!text.is_empty() && text.len() <= 19 && text.bytes().all(|byte| byte.is_ascii_digit()))
            .then(|| text.parse::<u64>().ok())
            .flatten()
            .filter(|value| *value <= i64::MAX.cast_unsigned())
            .ok_or_else(|| format!("{TIMING_METADATA_KEY}.{name} must fit nonnegative signed 63-bit nanoseconds"))?;
        Ok(value)
    };
    Ok(Some((read("start_ns")?, read("duration_ns")?)))
}

pub(crate) fn validate<'a>(
    gadgets: impl IntoIterator<Item = &'a bin::Gadget>,
    types: impl IntoIterator<Item = &'a bin::GadgetType>,
) -> Result<Option<ProgramTiming>, String> {
    let gadgets: Vec<_> = gadgets.into_iter().collect();
    if !gadgets.iter().any(|gadget| {
        gadget
            .metadata
            .as_ref()
            .is_some_and(|metadata| metadata.fields.contains_key(TIMING_METADATA_KEY))
    }) {
        return Ok(None);
    }
    let mut definitions = HashMap::new();
    for definition in types {
        if definitions.insert(definition.gtype, definition).is_some() {
            return Err(format!("duplicate gadget type {}", definition.gtype));
        }
    }
    let mut seen = HashMap::<u64, (&bin::GadgetType, u64)>::new();
    let mut next_gid = 1;
    let mut previous_completion = 0;
    let mut measurements = 0usize;
    let mut result = ProgramTiming {
        completions: Vec::with_capacity(gadgets.len()),
        cumulative_measurements: Vec::with_capacity(gadgets.len()),
    };
    for gadget in gadgets {
        let (start, duration) = gadget_timing(gadget)?.ok_or("mixed timed and untimed gadgets")?;
        let completion = start
            .checked_add(duration)
            .ok_or("gadget timing completion overflows uint64 nanoseconds")?;
        if completion > i64::MAX.cast_unsigned() {
            return Err("gadget completion exceeds signed 63-bit nanoseconds".into());
        }
        if completion < previous_completion {
            return Err("gadget completion times must be nondecreasing".into());
        }
        let definition = *definitions
            .get(&gadget.gtype)
            .ok_or_else(|| format!("unknown gadget type {}", gadget.gtype))?;
        if gadget.connectors.len() != definition.inputs.len() {
            return Err("gadget connector count does not match its input ports".into());
        }
        for (connector, input) in gadget.connectors.iter().zip(&definition.inputs) {
            let (source, source_completion) = seen
                .get(&connector.gid)
                .ok_or_else(|| format!("unknown or forward connector gid={}", connector.gid))?;
            let output = usize::try_from(connector.port)
                .ok()
                .and_then(|port| source.outputs.get(port))
                .ok_or("connector output port is out of range")?;
            if input.ptype != output.ptype {
                return Err("connector port types do not match".into());
            }
            if *source_completion > start {
                return Err("gadget starts before its input predecessor completes".into());
            }
        }
        if let Some(remote) = gadget
            .modifier
            .as_ref()
            .and_then(|modifier| modifier.remote_conditional_correction.as_ref())
        {
            for readout in &remote.remote_readouts {
                let (source, source_completion) = seen
                    .get(&readout.gid)
                    .ok_or_else(|| format!("unknown or forward readout gid={}", readout.gid))?;
                if usize::try_from(readout.readout_index).map_or(true, |index| index >= source.readouts.len()) {
                    return Err("remote readout index is out of range".into());
                }
                if *source_completion > start {
                    return Err("gadget starts before its remote readout predecessor completes".into());
                }
            }
        }
        let gid = if gadget.gid == 0 { next_gid } else { gadget.gid };
        if seen.insert(gid, (definition, completion)).is_some() {
            return Err(format!("duplicate gadget gid={gid}"));
        }
        next_gid = next_gid.max(gid.checked_add(1).ok_or("gadget ID overflow")?);
        measurements = measurements
            .checked_add(definition.measurements.len())
            .ok_or("program measurement count overflow")?;
        result.completions.push(Duration::from_nanos(completion));
        result.cumulative_measurements.push(measurements);
        previous_completion = completion;
    }
    Ok(Some(result))
}

pub(crate) fn validate_stim(timing: &ProgramTiming, delays: &[DelayBatch]) -> Result<(), String> {
    if delays.is_empty() {
        return Ok(());
    }
    let expected = canonical(
        timing
            .cumulative_measurements
            .iter()
            .copied()
            .zip(timing.completions.iter().copied()),
    );
    let actual = canonical(
        delays
            .iter()
            .map(|batch| (batch.cumulative_count, Duration::from_nanos(batch.delay_ns))),
    );
    if expected != actual {
        return Err("Stim #!delay schedule does not match simulation timing metadata".into());
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn timing_metadata(start_ns: u64, duration_ns: u64) -> Option<prost_types::Struct> {
    Some(prost_types::Struct {
        fields: [(
            TIMING_METADATA_KEY.to_owned(),
            prost_types::Value {
                kind: Some(Kind::StructValue(prost_types::Struct {
                    fields: [("start_ns", start_ns), ("duration_ns", duration_ns)]
                        .into_iter()
                        .map(|(key, value)| {
                            (
                                key.to_owned(),
                                prost_types::Value {
                                    kind: Some(Kind::StringValue(value.to_string())),
                                },
                            )
                        })
                        .collect(),
                })),
            },
        )]
        .into(),
    })
}

#[cfg(test)]
#[path = "../../tests/unit/program_timing_test.rs"]
mod metadata_tests;

fn canonical(batches: impl IntoIterator<Item = (usize, Duration)>) -> Vec<(usize, Duration)> {
    let mut result: Vec<(usize, Duration)> = Vec::new();
    for (count, deadline) in batches {
        if count == 0 && deadline.is_zero() {
            continue;
        }
        if let Some(last) = result.last_mut()
            && last.1 == deadline
        {
            last.0 = count;
        } else {
            result.push((count, deadline));
        }
    }
    result
}

pub(crate) fn duration(seconds: f64) -> Result<Duration, String> {
    Duration::try_from_secs_f64(seconds).map_err(|_| "invalid or overflowing timing duration".into())
}

pub(crate) fn legacy_completions(ranges: &[(usize, usize)], delays: &[DelayBatch]) -> Result<Vec<Duration>, String> {
    if delays.is_empty() {
        return Ok(vec![Duration::ZERO; ranges.len()]);
    }
    let total = ranges.last().map_or(0, |&(start, count)| start + count);
    let mut previous_count = 0;
    let mut previous_deadline = Duration::ZERO;
    for batch in delays {
        let deadline = Duration::from_nanos(batch.delay_ns);
        if batch.cumulative_count < previous_count || batch.cumulative_count > total || deadline < previous_deadline {
            return Err("invalid chronological Stim measurement schedule".into());
        }
        if !ranges.iter().any(|&(start, count)| start + count == batch.cumulative_count) {
            return Err("Stim delay splits a gadget's measurements".into());
        }
        previous_count = batch.cumulative_count;
        previous_deadline = deadline;
    }
    if previous_count != total {
        return Err("Stim schedule does not consume every program measurement".into());
    }
    ranges
        .iter()
        .map(|&(start, count)| {
            let batch = if count == 0 && start == total {
                delays.last()
            } else {
                delays.iter().find(|batch| batch.cumulative_count >= start + count)
            }
            .ok_or("Stim schedule has no deadline for this gadget")?;
            Ok(Duration::from_nanos(batch.delay_ns))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::stim_delays::extract_delay_schedule;

    #[test]
    fn compares_zero_measurement_and_terminal_only_timing() {
        for (counts, text) in [
            (
                vec![1, 1, 2, 2],
                "#!delay 0.01\nM 0\n#!delay 0.01\n#!delay 0.01\nM 0\n#!delay 0.01",
            ),
            (vec![0, 0, 0, 0], "#!delay 0.01\n#!delay 0.01\n#!delay 0.01\n#!delay 0.01"),
        ] {
            let total = *counts.last().unwrap();
            let timing = ProgramTiming {
                completions: (1..=4).map(|index| Duration::from_millis(index * 10)).collect(),
                cumulative_measurements: counts,
            };
            let delays = extract_delay_schedule(text, total);
            validate_stim(&timing, &delays).unwrap();
            validate_stim(&timing, &[]).unwrap();
            let mut mismatch = delays.clone();
            mismatch[1].delay_ns += 1;
            assert!(validate_stim(&timing, &mismatch).is_err());
            mismatch = delays;
            mismatch.pop();
            assert!(validate_stim(&timing, &mismatch).is_err());
        }
    }

    #[test]
    fn legacy_schedule_rejects_missing_batches_split_gadgets_and_invalid_times() {
        let ranges = [(0, 2), (2, 2)];
        for delays in [
            vec![DelayBatch {
                cumulative_count: 2,
                delay_ns: 100_000_000,
            }],
            vec![
                DelayBatch {
                    cumulative_count: 1,
                    delay_ns: 100_000_000,
                },
                DelayBatch {
                    cumulative_count: 4,
                    delay_ns: 200_000_000,
                },
            ],
            vec![
                DelayBatch {
                    cumulative_count: 2,
                    delay_ns: 200_000_000,
                },
                DelayBatch {
                    cumulative_count: 4,
                    delay_ns: 100_000_000,
                },
            ],
            vec![DelayBatch {
                cumulative_count: 5,
                delay_ns: 100_000_000,
            }],
        ] {
            assert!(legacy_completions(&ranges, &delays).is_err());
        }
    }

    #[test]
    fn exact_large_deadlines_do_not_round_trip_through_float() {
        let completion_ns = 16_777_216_000_000_001;
        let timing = ProgramTiming {
            cumulative_measurements: vec![0],
            completions: vec![Duration::from_nanos(completion_ns)],
        };
        let delays = extract_delay_schedule("#!delay 16777216.000000001", 0);
        assert_eq!(delays[0].delay_ns, completion_ns);
        validate_stim(&timing, &delays).unwrap();
        let different = extract_delay_schedule("#!delay 16777216.000000002", 0);
        assert!(validate_stim(&timing, &different).is_err());
    }

    #[test]
    fn long_zero_measurement_program_has_no_accumulated_float_error() {
        const GADGETS: usize = 18_000;
        let timing = ProgramTiming {
            cumulative_measurements: vec![0; GADGETS],
            completions: (1..=GADGETS)
                .map(|index| Duration::from_millis(u64::try_from(index).unwrap() * 100))
                .collect(),
        };
        let delays = extract_delay_schedule(&"#!delay 0.1\n".repeat(GADGETS), 0);
        assert_eq!(delays.last().unwrap().delay_ns, 1_800_000_000_000);
        validate_stim(&timing, &delays).unwrap();
    }
}
