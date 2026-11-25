use crate::{bin, util};

#[derive(Debug)]
pub(super) enum LinearMap {
    Zero,
    Copy(u128),
    Columns(Vec<(usize, u128)>),
}

impl LinearMap {
    pub(super) fn compile(
        source: Option<&util::BitMatrix>,
        modifier: Option<&bin::BitMatrixModifier>,
        rows: usize,
        columns: usize,
    ) -> Result<Self, String> {
        if rows > 128 {
            return Err("compiled frames support at most 128 output bits".into());
        }
        let parse = |matrix: &util::BitMatrix| -> Result<Vec<u128>, String> {
            let mut masks = vec![0_u128; columns];
            if usize::try_from(matrix.rows).ok() != Some(rows)
                || usize::try_from(matrix.cols).ok() != Some(columns)
                || matrix.i.len() != matrix.j.len()
            {
                return Err("frame matrix shape does not match its gadget".into());
            }
            for (&row, &column) in matrix.i.iter().zip(&matrix.j) {
                let row = usize::try_from(row).map_err(|_| "matrix row exceeds usize")?;
                let column = usize::try_from(column).map_err(|_| "matrix column exceeds usize")?;
                if row >= rows || column >= columns {
                    return Err("frame matrix index is out of range".into());
                }
                masks[column] |= 1_u128 << row;
            }
            Ok(masks)
        };
        let mut masks = if let Some(source) = source {
            parse(source)?
        } else {
            if rows != 0 && columns != 0 {
                return Err("missing nonempty frame matrix".into());
            }
            vec![0; columns]
        };
        if let Some(toggle) = modifier.and_then(|value| value.toggle.as_ref()) {
            for (target, value) in masks.iter_mut().zip(parse(toggle)?) {
                *target ^= value;
            }
        }
        if let Some(overwrite) = modifier.and_then(|value| value.overwrite.as_ref()) {
            masks = parse(overwrite)?;
        }
        let entries: Vec<_> = masks.into_iter().enumerate().filter(|(_, mask)| *mask != 0).collect();
        if entries.is_empty() {
            Ok(Self::Zero)
        } else if rows <= columns
            && entries.len() == rows
            && entries
                .iter()
                .enumerate()
                .all(|(index, &(column, mask))| column == index && mask == 1_u128 << index)
        {
            Ok(Self::Copy(bit_mask(rows)))
        } else {
            Ok(Self::Columns(entries))
        }
    }

    pub(super) fn apply(&self, input: u128) -> u128 {
        match self {
            Self::Zero => 0,
            Self::Copy(mask) => input & mask,
            Self::Columns(columns) => columns.iter().fold(0, |result, &(column, mask)| {
                if input & (1_u128 << column) != 0 {
                    result ^ mask
                } else {
                    result
                }
            }),
        }
    }

    pub(super) fn apply_raw(&self, bit: impl Fn(usize) -> bool) -> u128 {
        match self {
            Self::Zero => 0,
            Self::Copy(mask) => {
                let mut result = 0;
                for column in 0..mask.count_ones() {
                    if bit(column as usize) {
                        result |= 1_u128 << column;
                    }
                }
                result
            }
            Self::Columns(columns) => columns
                .iter()
                .fold(0, |result, &(column, mask)| if bit(column) { result ^ mask } else { result }),
        }
    }
}

pub(super) fn bit_mask(bits: usize) -> u128 {
    if bits == 128 { u128::MAX } else { (1_u128 << bits) - 1 }
}

pub(super) struct FrameType {
    pub input_bits: usize,
    pub output_bits: usize,
    pub readouts: Vec<Vec<usize>>,
    pub input_to_output: LinearMap,
    pub input_to_readout: LinearMap,
    pub readout_to_output: LinearMap,
    pub physical_to_output: LinearMap,
    pub remote_to_output: LinearMap,
}

impl FrameType {
    pub(super) fn compile(
        kind: &bin::GadgetType,
        modifier: Option<&bin::GadgetModifier>,
        input_bits: usize,
        output_bits: usize,
    ) -> Result<Self, String> {
        if input_bits >= 128 || output_bits > 128 || kind.readouts.len() > 128 {
            return Err("compiled-window supports at most 127 input and 128 output/readout frame bits".into());
        }
        let remote = modifier.and_then(|value| value.remote_conditional_correction.as_ref());
        if remote.is_some_and(|value| value.remote_readouts.len() > 128) {
            return Err("compiled-window supports at most 128 remote readout operands".into());
        }
        let mut readouts = Vec::new();
        for readout in &kind.readouts {
            let mut indices = Vec::new();
            for &index in &readout.measurement_indices {
                let index = usize::try_from(index).map_err(|_| "readout index exceeds usize")?;
                if index >= kind.measurements.len() {
                    return Err("readout measurement is out of range".into());
                }
                indices.push(index);
            }
            readouts.push(indices);
        }
        Ok(Self {
            input_bits,
            output_bits,
            readouts,
            input_to_output: LinearMap::compile(
                kind.correction_propagation.as_ref(),
                modifier.and_then(|value| value.correction_propagation_mod.as_ref()),
                output_bits,
                input_bits + 1,
            )?,
            input_to_readout: LinearMap::compile(
                kind.readout_propagation.as_ref(),
                modifier.and_then(|value| value.readout_propagation_mod.as_ref()),
                kind.readouts.len(),
                input_bits + 1,
            )?,
            readout_to_output: LinearMap::compile(
                kind.logical_correction.as_ref(),
                modifier.and_then(|value| value.logical_correction_mod.as_ref()),
                output_bits,
                kind.readouts.len(),
            )?,
            physical_to_output: LinearMap::compile(
                kind.physical_correction.as_ref(),
                modifier.and_then(|value| value.physical_correction_mod.as_ref()),
                output_bits,
                kind.measurements.len(),
            )?,
            remote_to_output: if let Some(remote) = remote {
                LinearMap::compile(remote.correction.as_ref(), None, output_bits, remote.remote_readouts.len())?
            } else {
                LinearMap::Zero
            },
        })
    }
}
